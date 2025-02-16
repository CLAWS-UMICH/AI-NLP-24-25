from flask import Flask, render_template, current_app
from flask_socketio import SocketIO, emit
from finetuned_models.agent import Agent, Tool
from dotenv import load_dotenv  
from time import sleep
load_dotenv()

socketio = SocketIO()



def create_app():
    app = Flask(__name__)
    socketio.init_app(app, async_mode='threading')
    return app

@socketio.on('connect')
def handle_connect():
    current_app.logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    

    
def ClickTasks():
    # current_app.logger.info("Clicking tasks...")
    socketio.emit('tool_response', {'tool_name': "ClickTasks"})
    return {"opened": True}

def ClickNavigation():
    # current_app.logger.info("Clicking navigation...")
    socketio.emit('tool_response', {'tool_name': "ClickNavigation"})
    return {"opened": True}

def ClickMessages():
    # current_app.logger.info("Clicking messages...")
    socketio.emit('tool_response', {'tool_name': "ClickMessages"})
    return {"opened": True}

def ClickSamples():
    # current_app.logger.info("Clicking samples...")
    socketio.emit('tool_response', {'tool_name': "ClickSamples"})
    return {"opened": True}

clickedVitals = False

@socketio.on('client_clicked_vitals')
def clickedVitalsCallback(data):
    global clickedVitals
    clickedVitals = True

def ClickVitals():
    # current_app.logger.info("Clicking vitals...")
    socketio.emit('tool_response', {'tool_name': "ClickVitals"})
    global clickedVitals
    while not clickedVitals:
        sleep(1)
    clickedVitals = False
    return {"opened": True}

@socketio.on('message')
def handle_message(data):
    current_app.logger.info(f'Received message: {data}')
    response = agent.ask(data)
    emit('response', {'data': response})
    
agent = Agent([
        Tool(
            name="ClickTasks",
            description="Opens the main task list interface.",
            params=[],
            return_description="True if successful, False otherwise",
            function=ClickTasks
        ),
        Tool(
            name="ClickNavigation", 

            description="Opens the navigation interface.",
            params=[],
            return_description="True if successful, False otherwise",
            function=ClickNavigation

        ),
        Tool(
            name="ClickMessages", 
            description="Opens the messages interface.",
            params=[],
            return_description="True if successful, False otherwise",
            function=ClickMessages

        ),
        Tool(
            name="ClickSamples", 
            description="Opens the samples interface.",
            params=[],
            return_description="True if successful, False otherwise",
            function=ClickSamples

        ),
        Tool(
            name="ClickVitals", 
            description="Opens the vitals interface.",
            params=[],
            return_description="True if successful, False otherwise",
            function=ClickVitals

        )
    ])
if __name__ == '__main__':
    # agent.clear_conversation_history()
    # agent.ask("get vitals")
    app = create_app()
    socketio.run(app, debug=True, port=5001, allow_unsafe_werkzeug=True)