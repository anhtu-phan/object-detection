import os
from flask import Flask, request, render_template, redirect, url_for
import argparse

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def index_post():
    file = request.files['image']
    file_save_path = os.path.join('static/input_images', file.filename)
    file.save(file_save_path)
    # img = cv2.imread(os.path.join('static/input_images', file.filename))

    return render_template('index.html', input_file=file.filename)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename="input_images/"+filename), code=301)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer text recognition demo")
    parser.add_argument("--port", default=9595)

    args = parser.parse_args()
    run_port = args.port

    if not os.path.exists('static/input_images'):
        os.makedirs('static/input_images')
    app.run('0.0.0.0', port=run_port)
