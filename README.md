## Tutor Bot

### Goal
Agent does the presentation (keeps going through slides in a loop) while also allowing questions.

### Acceptance Criteria
- Should smoothly return back to topic once presentation is finished
- Should reliably end the presentation when last slide is finished and enter QnA mode, i.e., simple 2 way conversation.

### Future Goal
- Should allow pauses sent from frontend

### Local Setup
#### Setup Agent
Environment file (.env at root)
```shell
# .env file
OPENAI_API_KEY=
```
Starting Python Agent
```shell
uv sync
python main.py
```
Starting Frontend
```shell
cd frontend
yarn install
yarn dev
```