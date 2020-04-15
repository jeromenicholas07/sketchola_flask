
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request, Response
import json
import numpy as np

app = Flask(__name__)

@app.route('/api/process', methods=['POST'])
def hello_world():
    xx = np.array(json.loads(request.form.get('data')))
    xx = np.reshape(xx, (1,1,256,256))
    
    return 'Hello from Flask2! '+ str(len(xx[0]))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
