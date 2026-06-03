import os
import glob
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Path to the logs directory
LOGS_DIR = r"C:\Users\rudra\.gemini\antigravity-ide\brain\6e292c00-5d5d-476c-978c-b60dcbd7eab6\.system_generated\tasks"
PORT = 8080

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aeras - Task Logs Viewer</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --surface: #1e293b;
            --text: #e2e8f0;
            --primary: #3b82f6;
            --accent: #10b981;
            --border: #334155;
        }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text);
            margin: 0;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        .sidebar {
            width: 300px;
            background-color: var(--surface);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 20px;
        }
        .sidebar h2 {
            margin-top: 0;
            font-size: 1.2rem;
            color: var(--primary);
            margin-bottom: 20px;
        }
        .task-list {
            list-style: none;
            padding: 0;
            margin: 0;
            overflow-y: auto;
            flex-grow: 1;
        }
        .task-item {
            padding: 12px 15px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid transparent;
        }
        .task-item:hover {
            background: rgba(255, 255, 255, 0.08);
        }
        .task-item.active {
            border-color: var(--primary);
            background: rgba(59, 130, 246, 0.1);
        }
        .task-name {
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 4px;
        }
        .task-id {
            font-size: 0.75rem;
            color: #94a3b8;
            font-family: monospace;
        }
        .main {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0;
            font-size: 1.5rem;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.1);
            color: var(--accent);
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--accent);
            margin-right: 6px;
            box-shadow: 0 0 8px var(--accent);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; box-shadow: 0 0 8px var(--accent); }
            50% { opacity: 0.5; box-shadow: 0 0 2px var(--accent); }
            100% { opacity: 1; box-shadow: 0 0 8px var(--accent); }
        }
        .log-container {
            flex-grow: 1;
            background-color: #000;
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 20px;
            overflow-y: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.85rem;
            line-height: 1.5;
            color: #a3be8c;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
            white-space: pre-wrap;
            scroll-behavior: smooth;
        }
        .auto-scroll-toggle {
            display: flex;
            align-items: center;
            font-size: 0.85rem;
            color: #94a3b8;
            cursor: pointer;
        }
        .auto-scroll-toggle input {
            margin-right: 8px;
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>Aeras Background Tasks</h2>
        <ul class="task-list" id="taskList">
            <!-- Populated via JS -->
        </ul>
    </div>
    <div class="main">
        <div class="header">
            <h1 id="currentTaskTitle">Select a task</h1>
            <div style="display: flex; gap: 20px; align-items: center;">
                <label class="auto-scroll-toggle">
                    <input type="checkbox" id="autoScroll" checked> Auto-scroll
                </label>
                <div class="status-badge" id="statusBadge" style="display: none;">
                    <div class="status-dot"></div> Polling Live
                </div>
            </div>
        </div>
        <div class="log-container" id="logContainer">
            Select a task from the sidebar to view its logs...
        </div>
    </div>

    <script>
        let currentFile = null;
        let pollInterval = null;
        let autoScroll = true;

        const knownTasks = {
            "task-536": "Phase 1: Model Training",
            "task-584": "Phase 2: ERA5 Download",
            "task-486": "Data Preprocessing"
        };

        document.getElementById('autoScroll').addEventListener('change', (e) => {
            autoScroll = e.target.checked;
        });

        async function fetchTasks() {
            try {
                const res = await fetch('/api/tasks');
                const files = await res.json();
                const list = document.getElementById('taskList');
                list.innerHTML = '';
                
                files.reverse().forEach(file => {
                    const taskId = file.replace('.log', '');
                    const friendlyName = knownTasks[taskId] || "Background Task";
                    
                    const li = document.createElement('li');
                    li.className = `task-item ${currentFile === file ? 'active' : ''}`;
                    li.innerHTML = `
                        <div class="task-name">${friendlyName}</div>
                        <div class="task-id">${taskId}</div>
                    `;
                    li.onclick = () => selectTask(file, friendlyName, li);
                    list.appendChild(li);
                });
            } catch (err) {
                console.error("Failed to fetch tasks", err);
            }
        }

        function selectTask(filename, title, element) {
            currentFile = filename;
            document.querySelectorAll('.task-item').forEach(el => el.classList.remove('active'));
            if(element) element.classList.add('active');
            
            document.getElementById('currentTaskTitle').innerText = title;
            document.getElementById('statusBadge').style.display = 'inline-flex';
            
            if (pollInterval) clearInterval(pollInterval);
            fetchLog(); // fetch immediately
            pollInterval = setInterval(fetchLog, 1500);
        }

        async function fetchLog() {
            if (!currentFile) return;
            try {
                const res = await fetch(`/api/logs?file=${currentFile}`);
                const text = await res.text();
                const container = document.getElementById('logContainer');
                
                // Avoid jarring DOM updates if nothing changed
                if (container.textContent !== text) {
                    container.textContent = text;
                    if (autoScroll) {
                        container.scrollTop = container.scrollHeight;
                    }
                }
            } catch (err) {
                console.error("Failed to fetch logs", err);
            }
        }

        fetchTasks();
    </script>
</body>
</html>
"""

class LogViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass # Suppress HTTP logs to keep terminal clean

    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
            
        elif parsed_path.path == '/api/tasks':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if not os.path.exists(LOGS_DIR):
                self.wfile.write(b'[]')
                return
                
            files = [f for f in os.listdir(LOGS_DIR) if f.endswith('.log')]
            # Sort by creation time
            files.sort(key=lambda x: os.path.getmtime(os.path.join(LOGS_DIR, x)))
            self.wfile.write(json.dumps(files).encode('utf-8'))
            
        elif parsed_path.path == '/api/logs':
            query = parse_qs(parsed_path.query)
            filename = query.get('file', [''])[0]
            
            if not filename or not filename.endswith('.log') or '/' in filename or '\\' in filename:
                self.send_response(400)
                self.end_headers()
                return
                
            filepath = os.path.join(LOGS_DIR, filename)
            
            if not os.path.exists(filepath):
                self.send_response(404)
                self.end_headers()
                return
                
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode('utf-8'))
            except Exception as e:
                self.wfile.write(str(e).encode('utf-8'))
                
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    print(f"Starting Aeras Log Viewer UI...")
    print(f"--> Open http://localhost:{PORT} in your browser to view live logs!")
    server = HTTPServer(('localhost', PORT), LogViewerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
