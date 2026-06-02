"""
Atmospheric chemistry module for coupled NO2/O3 photochemistry.

Implements the simplified troposphere chemistry:
    NO2 + hν → NO + O      (rate J(t), photolysis)
    O + O2 + M → O3
    NO + O3 → NO2 + O2     (rate k, titration)

[NO] is eliminated via the Leighton photochemical steady-state assumption.
"""

import math
import torch
import torch.nn as nn

# Delhi latitude (degrees North)
DELHI_LATITUDE = 28.6

# IST timezone offset from UTC (hours). Delhi = UTC + 5:30.
IST_OFFSET_HOURS = 5.5

# Time constants
SEC_PER_DAY = 86400.0
SEC_PER_YEAR = 31557600.0  # 365.25 days


def diurnal_photolysis(
    t_norm: torch.Tensor,
    t_min_unix: float,
    t_range_sec: float,
    lat_deg: float = DELHI_LATITUDE,
    tz_offset_hours: float = IST_OFFSET_HOURS,
) -> torch.Tensor:
    """
    Differentiable solar-zenith-angle cosine. Peaks at local noon, zero at night.

    Uses torch.remainder for periodicity — gradient is 1 almost everywhere,
    so autograd works through this function and through PINN time derivatives.

    Args:
        t_norm: tensor in [0, 1], normalized timestamp
        t_min_unix: real Unix timestamp at t_norm=0 (UTC seconds since epoch)
        t_range_sec: total time span in seconds
        lat_deg: latitude in degrees North
        tz_offset_hours: hours to add to UTC for local time (Delhi IST = +5.5)

    Returns:
        Tensor of same shape as t_norm, in [0, 1], representing cos(solar zenith)
        clamped at zero.
    """
    # Seconds since t_min (positive, max ~1.5e8 for 5 years)
    secs_since_min = t_norm * t_range_sec

    # Recover absolute time-of-day, using modulo to keep angles small.
    # t_min_unix offset into its own day:
    t_min_day_offset = t_min_unix % SEC_PER_DAY  # constant, 0–86400
    # Absolute seconds-into-UTC-day:
    secs_utc_day = torch.remainder(secs_since_min + t_min_day_offset, SEC_PER_DAY)
    # Apply timezone offset for local solar time:
    secs_local_day = torch.remainder(
        secs_utc_day + tz_offset_hours * 3600.0, SEC_PER_DAY
    )

    # Hour angle: peaks at zero at local solar noon (12:00)
    # h_angle = (2π · sec_local / 86400) - π
    h_angle = 2.0 * math.pi * secs_local_day / SEC_PER_DAY - math.pi

    # Day-of-year (fractional). Use modulo on accumulated seconds to keep precision.
    t_min_year_offset = t_min_unix % SEC_PER_YEAR
    secs_into_year = torch.remainder(secs_since_min + t_min_year_offset, SEC_PER_YEAR)
    day_of_year = secs_into_year / SEC_PER_DAY  # 0–365.25

    # Solar declination (radians)
    yearly_phase = 2.0 * math.pi * (day_of_year + 10.0) / 365.25
    decl = math.radians(-23.45) * torch.cos(yearly_phase)

    # Cosine of solar zenith angle
    lat_rad = math.radians(lat_deg)
    cos_zen = (
        math.sin(lat_rad) * torch.sin(decl)
        + math.cos(lat_rad) * torch.cos(decl) * torch.cos(h_angle)
    )

    return torch.clamp(cos_zen, min=0.0)


class ChemistryModule(nn.Module):
    """
    Differentiable atmospheric chemistry for NO2-O3 coupling.

    Holds two learnable scalars:
        log_J_amp        — amplitude of NO2 photolysis rate (normalized units)
        log_leighton_eps — Leighton ε regularizer (numerical stability)
    """

    def __init__(
        self,
        t_min_unix: float,
        t_range_sec: float,
        lat_deg: float = DELHI_LATITUDE,
        tz_offset_hours: float = IST_OFFSET_HOURS,
        log_J_amp_init: float = -1.0,
        log_leighton_eps_init: float = -3.0,
    ):
        super().__init__()
        # Store as floats — NOT buffers, since they're constants for the model life.
        self.t_min_unix = float(t_min_unix)
        self.t_range_sec = float(t_range_sec)
        self.lat_deg = float(lat_deg)
        self.tz_offset_hours = float(tz_offset_hours)

        self.log_J_amp = nn.Parameter(torch.tensor(float(log_J_amp_init)))
        self.log_leighton_eps = nn.Parameter(torch.tensor(float(log_leighton_eps_init)))

    @property
    def J_amp(self) -> torch.Tensor:
        return torch.exp(self.log_J_amp)

    @property
    def leighton_eps(self) -> torch.Tensor:
        return torch.exp(self.log_leighton_eps)

    def photolysis(self, t_norm: torch.Tensor) -> torch.Tensor:
        """J(t) for NO2 photolysis. Returns same shape as t_norm."""
        diurnal = diurnal_photolysis(
            t_norm, self.t_min_unix, self.t_range_sec,
            self.lat_deg, self.tz_offset_hours,
        )
        return self.J_amp * diurnal

    def chemistry_terms(
        self, t_norm: torch.Tensor, C_NO2: torch.Tensor, C_O3: torch.Tensor
    ):
        """
        Compute chemistry source/sink terms for NO2 and O3 PDEs.

        Both inputs are clamped to >= 0 to avoid unphysical negative concentrations
        during early training (network output is unconstrained).
        """
        J = self.photolysis(t_norm)
        eps = self.leighton_eps

        C_NO2_safe = torch.clamp(C_NO2, min=0.0)
        C_O3_safe = torch.clamp(C_O3, min=0.0)

        # Leighton ratio
        R = C_O3_safe / (C_O3_safe + eps)

        # Net chemistry (deviation from steady state)
        net = J * C_NO2_safe * (1.0 - R)

        return -net, net  # (chem_NO2, chem_O3)
