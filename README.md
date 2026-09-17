## Tutor Bot
Tutor bot is an MVP for an AI agent which teaches a class of students about natural disasters.

### Acceptance Criteria
Update existing agent code in such a way that following goals are met, while keeping conversation human-like and safe for students.

### Goals
- Should smoothly return back to topic once questions are answered.
- Should reliably end the presentation when last slide is finished and enter QnA mode, i.e., simple 2 way conversation, while also being able to return back at any point in presentation based on user request.
- Should allow pauses sent from frontend, i.e., agent stops speaking. On resuming, agent starts speaking exactly where it left off.
- Should have guard rails w.r.t underage users.
- Should have deterministic tests for all the business logic.
- Should have eval-style tests, i.e., LLM-as-a-judge tests.
- Should have a reliable way to ingest additional knowledge so that agent can answer questions better.
- Should log a metric report with information such as average latency, tokens consumed, etc. on console after user disconnects.

### Open Ended Goals
Architecturally, there must exist a way where agent's own old transcripts can be used to improve it's behavior.

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
