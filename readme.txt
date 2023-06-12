It is highly recommended to set up a virtual environment before running the project.

After cd ing into the project directory, type the following commands in the terminal:

python3 -m venv .venv
.\.venv\Scripts\activate

Now, intall the required libraries using the following commands:

pip install Flask
pip install numpy
pip install pandas
pip install pickle

Finally, type the following command to launch the web app:

python app.py

This will launch a local server and run the app at http://127.0.0.1:5000