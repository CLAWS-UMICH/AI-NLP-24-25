from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from betterlangchain.better import agent


socketio = SocketIO()

def create_app():
    app = Flask(__name__)
    socketio.init_app(app)
    return app

@socketio.on('connect')
def handle_connect():
    app.logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    
@socketio.on('message')
def handle_message(data):
    app.logger.info(f'Received message: {data}')
    response = agent.ask(data)
    emit('response', {'data': response})
    
if __name__ == '__main__':
    app = create_app()
    socketio.run(app, debug=True, port=5001)