from flask import Flask, jsonify, request
import pickle
import numpy as np
from flask_cors import CORS 
import json
import pandas as pd
from surprise import SVD, Dataset, Reader

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) # cho phep truy cap tu client

popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
data = pickle.load(open('data.pkl', 'rb'))
# books = pickle.load(open('books.pkl', 'rb'))
# similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))
similarity_matrix = pickle.load(open('similarity_matrix.pkl', 'rb'))  # Đọc ma trận tương đồng mới

def recommend1(book_name, data):
    # Lấy chỉ mục của cuốn sách đầu vào
    index = data[data['bookTitle'] == book_name].index[0]
    
    # Sắp xếp các cuốn sách khác theo độ tương đồng giảm dần
    similar_items = sorted(list(enumerate(similarity_matrix[index])), key=lambda x: x[1], reverse=True)[1:11]
    
    recommendations1 = []
    for i in similar_items:
        # item = data.iloc[i[0]].to_list()  # Sử dụng phương thức to_list()
        item = data.iloc[i[0]].to_dict()  # Chuyển đổi dữ liệu thành từ điển
        recommendations1.append(item)

    return recommendations1

@app.route('/api/popular-books')
def get_popular_books():
    data = popular_df.to_dict(orient='records')
    return jsonify(data)

# @app.route('/api/recommend', methods=['POST'])
# def recommend_books():
#     user_input = request.json.get('user_input')
#     index = np.where(pt.index == user_input)[0][0]
#     similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:11]
#     data = []
#     for i in similar_items:
#         item = []
#         temp_df = books[books['Book-Title'] == pt.index[i[0]]]
#         item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
#         item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
#         item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
#         data.append(item)
#     return jsonify(data)

@app.route('/api/recommend', methods=['POST'])
def recommend_books():
    user_input = request.json.get('user_input')
    
    # Kiểm tra xem cuốn sách đầu vào có tồn tại trong dữ liệu hay không
    if user_input in data['bookTitle'].values:
        recommendations = recommend1(user_input, data)
        if isinstance(recommendations, str):
            try:
                recommendations = json.loads(recommendations)
            except json.JSONDecodeError:
                recommendations = []
        
        # Đảm bảo recommendations là một list
        if not isinstance(recommendations, list):
            recommendations = [recommendations] if recommendations else []
        # print("Type of recommendations after processing:", type(recommendations))
        return jsonify(recommendations)
    else:
        # Nếu cuốn sách đầu vào không tồn tại, trả về danh sách trống
        return jsonify([])

# MODEL SVD
def load_model():
    with open('svd_model.pkl', 'rb') as f:
        model, trainset = pickle.load(f)
    return model, trainset

model, trainset = load_model()

rating = pd.read_csv('cleaned_ratings.csv')

# Generate recommendations using the loaded model
def generate_recommendationsSVD(userID, get_recommend=10):
    
    # Generate predictions for all pairs (user, item) that are not in the training set
    testset = trainset.build_anti_testset()
    predictions = model.test(testset)
    predictions_df = pd.DataFrame(predictions)
    
    # Filter predictions for the given user and get the top recommendations
    predictions_userID = predictions_df[predictions_df['uid'] == userID].sort_values(by="est", ascending=False).head(get_recommend)
    # recommendations = list(predictions_userID['iid'])
    
    # Remove duplicate book recommendations
    predictions_userID = predictions_userID.drop_duplicates(subset=['iid']).head(get_recommend)
    
    # Merge with the original rating dataframe to get book titles and image URLs
    recommendations_df = predictions_userID.merge(rating, left_on='iid', right_on='Book-Title')[['Book-Title', 'Image-URL-M']]
    
     # Drop duplicates
    recommendations_df = recommendations_df.drop_duplicates(subset=['Book-Title'])

    # Ensure we only return the number of recommendations requested
    recommendations_df = recommendations_df.head(get_recommend)

    # Convert to a list of tuples (title, image_url)
    recommendations = list(recommendations_df.itertuples(index=False, name=None))

    return recommendations

@app.route('/api/recommendSVD', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)
    if user_id is None:
        return jsonify({'error': 'user_id parameter is required'}), 400
    
    recommendations = generate_recommendationsSVD(userID=user_id)
    
    return jsonify({'user_id': user_id, 'recommendations': recommendations})

# run in developer environment
# if __name__ == '__main__':
#     app.run(debug=True, port=5001)

# run in deployment environment
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5001)

# 123
#456