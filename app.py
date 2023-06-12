from flask import Flask, request
from flask import render_template, request, redirect, url_for, flash, session
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

app = Flask(__name__, template_folder='templates')


def create_raw_dataframe(filepath, columns):
    return pd.read_csv(filepath, names=columns, engine='python', delimiter='::', encoding='latin-1')

def create_movie_user_dataframe(num_rows, num_cols, data):
    dim = (num_rows, num_cols)
    scores_mat = np.ndarray(shape=dim)
    x = data.serial_number_movie.values - 1
    y = data.serial_number_user.values - 1
    # dtype=np.uint8
    scores_mat[x, y] = data.score.values
    return scores_mat

def normalize(matrix):
    mean = np.mean(matrix, 1)
    mean_matrix = np.asarray([(mean)])
    transposed_mean_matrix = mean_matrix.T
    return matrix - transposed_mean_matrix

def find_singular_value_decomposition(normal, scores_mat):
    transposed_normal = normal.T
    root_dim = np.sqrt(scores_mat.shape[0] - 1)
    matrix = transposed_normal / root_dim
    decomposition = np.linalg.svd(matrix)
    return decomposition

def find_cosimilarity(data, serial_number_movie):
    movie = data[serial_number_movie - 1]
    mod = np.sqrt(np.sum(data * data, axis=1))
    numerator = np.dot(movie, data.T)
    denominator = mod[serial_number_movie - 1] * mod
    similarity = numerator / denominator
    return similarity

def find_most_similar(similarity):
    return np.argsort(-similarity)[:8]   

def get_recommendations(movies, positions):
    recommendations = []
    for i in positions:
        movie_name = movies[movies.serial_number_movie == i + 1].name.values[0]
        # movie_genre = movies[movies.serial_number_movie == i + 1].genre.values[0]
        recommendations.append(movie_name)
        # + ' ' + movie_genre
    return recommendations    

def print_results(recommendations):
    for movie in recommendations:
        print(movie)  

def get_serial_number_from_name(movies, name):
    query_text = "name=='{}'".format(name)
    row = movies.query(query_text)["serial_number_movie"]
    return row.values[0]

def initialize(ratings_filepath, movies_filepath):
    scores_columns = ['serial_number_user', 'serial_number_movie', 'score', 'year']
    movies_columns = ['serial_number_movie', 'name', 'genre']
    ratings = create_raw_dataframe(ratings_filepath, scores_columns)
    movies = create_raw_dataframe(movies_filepath, movies_columns)
    return (ratings, movies)

def store_decomposition(U, S, V):
    decomp = {'U': U, 'S': S, 'V': V}
    with open('svd.pickle', 'wb') as file:
        pickle.dump(decomp, file) 


def work(name):
    ratings, movies = initialize('data/ratings.dat', 'data/movies.dat')

    num_rows = np.max(ratings.serial_number_movie.values)
    num_cols = np.max(ratings.serial_number_user.values)

    movie_user_matrix = create_movie_user_dataframe(num_rows, num_cols, ratings)
    normalized_movie_user_matrix = normalize(movie_user_matrix)

    file_path = Path('svd.pickle')

    if file_path.exists():
        with open('svd.pickle', 'rb') as file:
            decomp = pickle.load(file) 
            U, S, V = decomp['U'], decomp['S'], decomp['V']
    else:
        U, S, V = find_singular_value_decomposition(normalized_movie_user_matrix, movie_user_matrix)  
        store_decomposition(U, S, V)


    num_principal_components = 50
    reduced_dimensions_data = V.T[:, :num_principal_components]

    serial_number_movie = get_serial_number_from_name(movies, name)

    similarities = find_cosimilarity(reduced_dimensions_data, serial_number_movie)
    positions = find_most_similar(similarities)
    movies = get_recommendations(movies, positions)
    return movies[1:]
	#     print_results(movies)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie = request.form.get('movie')
        recommendations = work(movie) 
        result = "<br>".join([str(temp) for temp in recommendations])    
        return f'<div style="text-align:center; font-family: Helvetica, sans-serif;"><h1>Movie Recommender System</h1><p style="text-decoration: underline;">{movie}</p><p> {result}</p></div>'
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)





# from flask import Flask, request
# from flask import render_template, request, redirect, url_for, flash, session
# import numpy as np
# import pandas as pd
# import pickle
# from pathlib import Path
# import pickle

# app = Flask(__name__)

# def create_raw_dataframe(filepath, columns):
#     return pd.read_csv(filepath, names=columns, engine='python', delimiter='::', encoding='latin-1')

# def create_movie_user_dataframe(num_rows, num_cols, data):
#     dim = (num_rows, num_cols)
#     scores_mat = np.ndarray(shape=dim)
#     x = data.serial_number_movie.values - 1
#     y = data.serial_number_user.values - 1
#     # dtype=np.uint8
#     scores_mat[x, y] = data.score.values
#     return scores_mat

# def normalize(matrix):
#     mean = np.mean(matrix, 1)
#     mean_matrix = np.asarray([(mean)])
#     transposed_mean_matrix = mean_matrix.T
#     return matrix - transposed_mean_matrix

# def find_singular_value_decomposition(normal, scores_mat):
#     transposed_normal = normal.T
#     root_dim = np.sqrt(scores_mat.shape[0] - 1)
#     matrix = transposed_normal / root_dim
#     decomposition = np.linalg.svd(matrix)
#     return decomposition

# def find_cosimilarity(data, serial_number_movie):
#     movie = data[serial_number_movie - 1]
#     mod = np.sqrt(np.sum(data * data, axis=1))
#     numerator = np.dot(movie, data.T)
#     denominator = mod[serial_number_movie - 1] * mod
#     similarity = numerator / denominator
#     return similarity

# def find_most_similar(similarity):
#     return np.argsort(-similarity)[:8]   

# def get_recommendations(movies, positions):
#     recommendations = []
#     for i in positions:
#         movie_name = movies[movies.serial_number_movie == i + 1].name.values[0]
#         # movie_genre = movies[movies.serial_number_movie == i + 1].genre.values[0]
#         recommendations.append(movie_name)
#         # + ' ' + movie_genre
#     return recommendations    

# def print_results(recommendations):
#     for movie in recommendations:
#         print(movie)  

# def get_serial_number_from_name(movies, name):
#     query_text = "name=='{}'".format(name)
#     row = movies.query(query_text)["serial_number_movie"]
#     return row.values[0]

# def initialize(ratings_filepath, movies_filepath):
#     scores_columns = ['serial_number_user', 'serial_number_movie', 'score', 'year']
#     movies_columns = ['serial_number_movie', 'name', 'genre']
#     ratings = create_raw_dataframe(ratings_filepath, scores_columns)
#     movies = create_raw_dataframe(movies_filepath, movies_columns)
#     return (ratings, movies)

# def store_decomposition(U, S, V):
#     decomp = {'U': U, 'S': S, 'V': V}
#     with open('svd.pickle', 'wb') as file:
#         pickle.dump(decomp, file) 


# def work(name):
#     ratings, movies = initialize('data/ratings.dat', 'data/movies.dat')

#     num_rows = np.max(ratings.serial_number_movie.values)
#     num_cols = np.max(ratings.serial_number_user.values)

#     movie_user_matrix = create_movie_user_dataframe(num_rows, num_cols, ratings)
#     normalized_movie_user_matrix = normalize(movie_user_matrix)

#     file_path = Path('svd.pickle')

#     if file_path.exists():
#         with open('svd.pickle', 'rb') as file:
#             decomp = pickle.load(file) 
#             U, S, V = decomp['U'], decomp['S'], decomp['V']
#     else:
#         U, S, V = find_singular_value_decomposition(normalized_movie_user_matrix, movie_user_matrix)  
#         store_decomposition(U, S, V)


#     num_principal_components = 50
#     reduced_dimensions_data = V.T[:, :num_principal_components]

#     serial_number_movie = get_serial_number_from_name(movies, name)

#     similarities = find_cosimilarity(reduced_dimensions_data, serial_number_movie)
#     positions = find_most_similar(similarities)
#     movies = get_recommendations(movies, positions)
#     return movies[1:]
# 	#     print_results(movies)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         movie = request.form.get('movie')
#         recommendations = work(movie) 
#         result = "<br>".join([str(temp) for temp in recommendations])    
#         return render_template('recommendations.html', movie = movie, result=result)
#     return render_template('index.html')


# if __name__ == '__main__':
#     app.run()



# # from flask import Flask, jsonify

# # app = Flask(__name__)

# # @app.route('/api/data', methods=['GET'])
# # def get_data():
# #     data = {'message': 'Hello from Flask!'}
# #     return jsonify(data)

# # if __name__ == '__main__':
# #     app.run(debug=True)
