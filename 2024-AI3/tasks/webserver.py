# cloudflared tunnel --url http://localhost:3000

import http.server
import socketserver
import os

# Define the port and the directory to serve
PORT = 3000
DIRECTORY = "assets"

# Change the working directory to the assets folder
os.chdir(DIRECTORY)

# Create a handler to serve files
Handler = http.server.SimpleHTTPRequestHandler

# Start the server
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving files from '{DIRECTORY}' directory at http://localhost:{PORT}")
    httpd.serve_forever()