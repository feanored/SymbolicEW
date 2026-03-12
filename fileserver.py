import http.server
import socketserver
import os
import time
import html
from datetime import datetime

class HandlerComDetalhes(http.server.SimpleHTTPRequestHandler):
    def list_directory(self, path):
        try:
            listagem = os.listdir(path)
        except OSError:
            self.send_error(404, "No permission to list directory")
            return None
        
        listagem.sort(key=lambda a: a.lower())
        
        # Cria a página HTML
        f = io.BytesIO()
        displaypath = html.escape(self.path)
        f.write(b'<!DOCTYPE html><html><head><meta charset="utf-8">')
        f.write(f'<title>Diretorio: {displaypath}</title></head>'.encode('utf-8'))
        f.write(f'<body><h1>Diretorio: {displaypath}</h1>'.encode('utf-8'))
        f.write(b'<hr><table>')
        
        # Adiciona link para voltar
        if self.path != "/":
            f.write(b'<tr><td><a href="..">../</a></td></tr>')

        for name in listagem:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            
            # Pega atributos (tamanho e data)
            stats = os.stat(fullname)
            filesize = stats.st_size
            mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats.st_mtime))
            
            # Formata tamanho (bytes, KB, MB)
            if filesize < 1024:
                size_str = f"{filesize} B"
            elif filesize < 1024*1024:
                size_str = f"{filesize/1024:.1f} KB"
            else:
                size_str = f"{filesize/(1024*1024):.1f} MB"

            # Adiciona classe CSS para alinhar
            if os.path.isdir(fullname):
                displayname = name + "/"
                size_str = "-"
            
            f.write(f'<tr><td><a href="{html.escape(linkname)}">{html.escape(displayname)}</a></td>'.encode('utf-8'))
            f.write(f'<td><i>({size_str} - {mtime})</i></td></tr>'.encode('utf-8'))

        f.write(b'</table><hr></body></html>')
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return f

# Para resolver a importação do io no script acima
import io

# Configuração do Servidor
PORTA = 8000
Handler = HandlerComDetalhes

with socketserver.TCPServer(("", PORTA), Handler) as httpd:
    print(f"Servidor iniciado na porta {PORTA}")
    print(f"Acesse: http://localhost:{PORTA}")
    httpd.serve_forever()

