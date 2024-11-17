import sys
sys.path.insert(0, './langchain')
from main import langchain_agent_response
#from langchain.main import langchain_agent_response
from flask import Flask, request

app = Flask(__name__)


@app.route('/parsevc', methods =['POST'])
def submit_data():
   print(request.form)
   query = request.form['question']
   response = langchain_agent_response(query)
   return response





if __name__ == '__main__':
   app.run()