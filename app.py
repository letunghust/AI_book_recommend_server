from flask import Flask, jsonify, request
import pickle
import numpy as np
from flask_cors import CORS 

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
        return jsonify(recommendations)
    else:
        # Nếu cuốn sách đầu vào không tồn tại, trả về danh sách trống
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True, port=5001)

# if __name__ == "__main__":
#     from waitress import serve
#     serve(app, host="0.0.0.0", port=5001)