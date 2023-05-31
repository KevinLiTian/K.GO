# K.GO - A Deep Learning Go AI

![Home](/asset/home.png)

### Motivation
Personally as an amateur 5 dan Go player, I played the game of Go since I was 8. I stopped playing after going to high school, but I've always
been paying attention to the world of Go. In 2016, Alpha Go defeated Lee Sedol 4 - 1, shocked the world as well as myself. Since then, as a computer engineering
student, I've wanted to create my own Go AI, and as a metric, to defeat me.

### Opportunity
Going into my third year, I learned more about machine learning and artificial intelligence, and skilled with frameworks like PyTorch. I decided to
follow the Alpha Go published articles and build my own Go AI. And with some knowledge in web development, I also created the frontend with React
to be able to play against my own AI.

## How To Play

Install all necessary dependencies listed in `requirements.txt`.

```sh
pip install -r requirements.txt
```

Run `ai_play.py` to run the websocket and start a simple server to run the models.

```sh
python ai_play.py
```

Start up the frontend with `npm`, so make sure `npm` is installed in your device.

```sh
# Assume in the root directory
cd frontend
npm run dev
```

You should be able to see the web page, click on the `Play` button on the top right corner, select Human Vs. AI to play against the models.
Recommand to play against difficulties 1 or 2 since 3 uses MCTS algorithm and the response time could be significantly longer than 1 and 2 as they only use
the model to inference.
