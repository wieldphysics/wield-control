from flask import Flask, redirect, url_for, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/test")
def test_html():
    return render_template("test.html")

@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        user = request.form["nm"]
        return redirect(url_for("user", usr=user))
    else:
        return render_template("login.html")

@app.route("/user/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"

@app.route("/graph",)
def graph():
    return jsonify({
          'nodes': [
              {'data': { 'id': 'a', 'parent': 'b' }, 'position': { 'x': 215, 'y': 85 } },
              {'data': { 'id': 'b' } },
              {'data': { 'id': 'c', 'parent': 'b' }, 'position': { 'x': 300, 'y': 85 } },
              {'data': { 'id': 'd' }, 'position': { 'x': 215, 'y': 175 } },
              {'data': { 'id': 'e' } },
              {'data': { 'id': 'f', 'parent': 'e' }, 'position': { 'x': 300, 'y': 175 } }
          ],
          'edges': [
              { 'data': { 'id': 'ad', 'source': 'a', 'target': 'd' } },
              { 'data': { 'id': 'eb', 'source': 'e', 'target': 'b' } }

          ]
      })

@app.route("/graph_recv", methods=['POST'])
def graph_recv():
    print(request.get_json())
    return ""

if __name__ == "__main__":
    use_livereload = True
    if use_livereload:
        try:
            import livereload
        except ImportError:
            print("Would use livereload if it is installed")
            use_livereload = False

    if use_livereload:
        from packaging import version
        if version.parse(livereload.__version__) > version.parse('2.5.1'):
            print("Livereload version >2.5.1 has a bug with newer Tornado causing an exception on reload, https://github.com/lepture/python-livereload/issues/176, fix with 'pip install livereload==2.5.1' ")

        app.debug = True

        server = livereload.Server(app.wsgi_app)
        server.serve(port = 5000)
    else:
        app.run(debug=True)

