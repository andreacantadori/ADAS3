from http.server import BaseHTTPRequestHandler,HTTPServer
from threading import Thread

class myHandler(BaseHTTPRequestHandler):
    def do_GET(self):
         self.send_response(200)
         self.send_header('Content-type','text/html')
         self.end_headers()
         self.wfile.write('<body>'.encode() + self.message.encode() + '</body>'.encode())
         return
    def setMessage(self,msg):
        self.message = msg
        return

class intercom:
    def startServer(self):
        myHandler.setMessage(myHandler,' ')
        server = HTTPServer(('localhost', 8080), myHandler)
        thread = Thread(target=server.serve_forever, args=())
        thread.setDaemon(True)
        thread.start()
    def stopServer(self):
        server.socket.close()
    def sendOutput(self,msg):
        myHandler.setMessage(myHandler,msg)
    def getInput(self):
        pass