import http.server
import socketserver
import os

# Configurações
PORTA = 8002
DIRETORIO = "."  # Pasta atual
EXTENSOES = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

class HandlerImg(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            # Gera a lista de imagens
            files = sorted([f for f in os.listdir(DIRETORIO) if f.lower().endswith(EXTENSOES)])
            
            html = "<html><head><title>Visualizador</title></head><body>"
            html += "<h1>Galeria de Imagens</h1><div style='display:flex; flex-wrap:wrap;'>"
            
            for file in files:
                # Adiciona cada imagem como um link para ela mesma
                html += f"<div style='margin:10px;'>"
                html += f"<a href='{file}' target='_blank'>"
                html += f"<img src='{file}' style='max-height:150px; max-width:150px; border:1px solid #ccc;'>"
                html += f"</a><br><small>{file}</small></div>"
            
            html += "</div></body></html>"
            self.wfile.write(html.encode('utf-8'))
        else:
            # Serve as imagens normalmente
            super().do_GET()

# Inicia o servidor
with socketserver.TCPServer(("", PORTA), HandlerImg) as httpd:
    print(f"Servidor rodando em http://localhost:{PORTA}")
    print("Pressione Ctrl+C para parar.")
    httpd.serve_forever()
