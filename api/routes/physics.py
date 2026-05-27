from fastapi import APIRouter, HTTPException
from api.model_loader import ModelLoader

router = APIRouter()


@router.get("/")
def get_physics_params():
    model = ModelLoader.get_pinn()
    if model is None:
        raise HTTPException(503, "Model not loaded.")

    Dx  = model.Dx.item()
    Dy  = model.Dy.item()
    lam = model.lambda_dep.item()

    return {
        "Dx": round(Dx, 6),
        "Dy": round(Dy, 6),
        "lambda_dep": round(lam, 6),
        "log_Dx": round(model.log_Dx.item(), 4),
        "log_Dy": round(model.log_Dy.item(), 4),
        "log_lambda_dep": round(model.log_lambda_dep.item(), 4),
        "note": (
            "Dx, Dy: diffusion coefficients (normalized space²/time). "
            "lambda_dep: PM2.5 deposition/removal rate (1/normalized_time). "
            "All learned from data — not prescribed."
        ),
    }
