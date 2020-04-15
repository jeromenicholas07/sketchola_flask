
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request, Response
import json

app = Flask(__name__)

@app.route('/api/process', methods=['POST'])
def hello_world():
    xx = np.array(json.loads(request.form.get('data')))
    xx = np.reshape(xx, (1,1,256,256))
    
    return 'Hello from Flask2! '+ str(len(xx[0]))

