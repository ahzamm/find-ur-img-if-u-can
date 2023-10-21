from flask import Flask, request

app = Flask(__name__)

@app.route('/photos', methods=['POST'])
def upload_photos():
    file = request.files['image']
    file.save('uploads/' + file.filename)
    return 'File uploaded successfully!'

if __name__ == '__main__':
    app.run(debug=True)
