from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def gfg():
   return 'geeksforgeeks'

@app.route('/hello/<name>')
def hello_name(name):
   return 'Hello %s!' % name

@app.route('/parsevc', methods =['POST'])
def submit_data():
   data = request.form
   return data





if __name__ == '__main__':
   app.run()