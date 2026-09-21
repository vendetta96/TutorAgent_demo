/**
 * Tutor Bot frontend: connects to the Pipecat agent over WebSocket, shows the
 * lesson state pushed by the server, and drives pause / resume.
 *
 * Pause is two-layered: the server buffers the bot's output (authoritative), and
 * the browser suspends its AudioContext so already-received audio stops instantly
 * and resumes sample-exact.
 */

import { PipecatClient, type PipecatClientOptions } from '@pipecat-ai/client-js';
import { WavMediaManager, WebSocketTransport } from '@pipecat-ai/websocket-transport';

const SERVER = 'http://localhost:7860';

type LessonState = {
    type: 'state';
    mode: 'idle' | 'presenting' | 'qna' | 'ended';
    paused: boolean;
    slide_number: number;
    total_slides: number;
    slide_title: string | null;
    talking_points: string[];
    completed_slides: number[];
    bot_speaking?: boolean;
};

class PausableMediaManager extends WavMediaManager {
    private get audioContext(): AudioContext | undefined {
        // WavMediaManager keeps its player private; the context is what we need to freeze.
        return (this as any)._wavStreamPlayer?.context as AudioContext | undefined;
    }

    async pausePlayback(): Promise<void> {
        const ctx = this.audioContext;
        if (ctx && ctx.state === 'running') await ctx.suspend();
    }

    async resumePlayback(): Promise<void> {
        const ctx = this.audioContext;
        if (ctx && ctx.state === 'suspended') await ctx.resume();
    }
}

class TutorApp {
    private client: PipecatClient | null = null;
    private media: PausableMediaManager | null = null;
    private paused = false;
    private botLine: HTMLElement | null = null;

    private el = {
        connect: document.getElementById('connect-btn') as HTMLButtonElement,
        disconnect: document.getElementById('disconnect-btn') as HTMLButtonElement,
        pause: document.getElementById('pause-btn') as HTMLButtonElement,
        resume: document.getElementById('resume-btn') as HTMLButtonElement,
        status: document.getElementById('connection-status')!,
        dot: document.getElementById('status-dot')!,
        mode: document.getElementById('mode-badge')!,
        counter: document.getElementById('slide-counter')!,
        progress: document.getElementById('progress-bar')!,
        title: document.getElementById('slide-title')!,
        points: document.getElementById('talking-points')!,
        speaking: document.getElementById('speaking-indicator')!,
        pausedBanner: document.getElementById('paused-banner')!,
        transcript: document.getElementById('transcript')!,
        reportCard: document.getElementById('report-card')!,
        report: document.getElementById('report')!,
        debug: document.getElementById('debug-log')!,
    };

    constructor() {
        this.el.connect.addEventListener('click', () => this.connect());
        this.el.disconnect.addEventListener('click', () => this.disconnect());
        this.el.pause.addEventListener('click', () => this.pause());
        this.el.resume.addEventListener('click', () => this.resume());
    }

    // ---- UI helpers -------------------------------------------------------

    private log(message: string): void {
        const entry = document.createElement('div');
        entry.textContent = `${new Date().toLocaleTimeString()} ${message}`;
        this.el.debug.appendChild(entry);
        this.el.debug.scrollTop = this.el.debug.scrollHeight;
        console.log(message);
    }

    private setStatus(text: string, dot: 'off' | 'on' | 'warn'): void {
        this.el.status.textContent = text;
        this.el.dot.className = `dot dot-${dot}`;
    }

    private setButtons(connected: boolean): void {
        this.el.connect.disabled = connected;
        this.el.disconnect.disabled = !connected;
        this.el.pause.disabled = !connected || this.paused;
        this.el.resume.disabled = !connected || !this.paused;
    }

    private addTranscript(kind: 'user' | 'bot' | 'sys', text: string): HTMLElement {
        const line = document.createElement('div');
        line.className = `msg msg-${kind}`;
        line.textContent = text;
        this.el.transcript.appendChild(line);
        this.el.transcript.scrollTop = this.el.transcript.scrollHeight;
        return line;
    }

    private renderState(state: LessonState): void {
        const label = state.paused ? 'paused' : state.mode;
        this.el.mode.className = `badge badge-${label}`;
        this.el.mode.textContent = state.paused ? 'Paused' : ({ idle: 'Idle', presenting: 'Presenting', qna: 'Q&A', ended: 'Ended' } as const)[state.mode];

        if (state.slide_number > 0) {
            this.el.counter.textContent = `Slide ${state.slide_number} / ${state.total_slides}`;
            this.el.progress.style.width = `${(state.completed_slides.length / state.total_slides) * 100}%`;
            this.el.title.textContent = state.slide_title ?? '';
            this.el.points.replaceChildren(
                ...state.talking_points.map((p) => {
                    const li = document.createElement('li');
                    li.textContent = p;
                    return li;
                })
            );
        }
        if (state.mode === 'qna') {
            this.el.title.textContent = 'Q&A — ask anything, or ask to revisit a slide';
        } else if (state.mode === 'ended') {
            this.el.title.textContent = 'Lesson finished';
        }
        this.el.speaking.classList.toggle('hidden', !state.bot_speaking || state.paused);
        this.el.pausedBanner.classList.toggle('hidden', !state.paused);
        this.paused = state.paused;
        this.setButtons(this.client !== null);
    }

    // ---- connection -------------------------------------------------------

    async connect(): Promise<void> {
        this.el.reportCard.classList.add('hidden');
        this.el.transcript.replaceChildren();
        this.setStatus('Connecting…', 'warn');
        this.el.connect.disabled = true;

        this.media = new PausableMediaManager();
        const options: PipecatClientOptions = {
            transport: new WebSocketTransport({ mediaManager: this.media }),
            enableMic: true,
            enableCam: false,
            callbacks: {
                onConnected: () => {
                    this.setStatus('Connected', 'on');
                    this.setButtons(true);
                },
                onDisconnected: () => {
                    this.setStatus('Disconnected', 'off');
                    this.paused = false;
                    this.setButtons(false);
                    this.el.speaking.classList.add('hidden');
                    this.el.pausedBanner.classList.add('hidden');
                    this.client = null;
                    void this.showReport();
                },
                onBotReady: () => this.log('bot ready'),
                onServerMessage: (data: any) => {
                    if (data?.type === 'state') this.renderState(data as LessonState);
                },
                onUserTranscript: (data) => {
                    if (data.final) this.addTranscript('user', data.text);
                },
                onBotStartedSpeaking: () => {
                    this.botLine = null;
                    if (!this.paused) this.el.speaking.classList.remove('hidden');
                },
                onBotStoppedSpeaking: () => {
                    this.botLine = null;
                    this.el.speaking.classList.add('hidden');
                },
                onBotTtsText: (data) => {
                    if (!this.botLine) this.botLine = this.addTranscript('bot', '');
                    this.botLine.textContent = `${this.botLine.textContent} ${data.text}`.trim();
                    this.el.transcript.scrollTop = this.el.transcript.scrollHeight;
                },
                onError: (error) => this.log(`error: ${JSON.stringify(error)}`),
                onMessageError: (error) => this.log(`message error: ${JSON.stringify(error)}`),
            },
        };

        try {
            this.client = new PipecatClient(options);
            (window as any).pcClient = this.client;
            await this.client.initDevices();
            await this.client.startBotAndConnect({ endpoint: `${SERVER}/connect` });
            this.addTranscript('sys', 'Connected. The lesson will begin in a moment.');
        } catch (error) {
            this.log(`connect failed: ${(error as Error).message}`);
            this.setStatus('Error', 'off');
            this.setButtons(false);
            this.client = null;
        }
    }

    async disconnect(): Promise<void> {
        if (!this.client) return;
        try {
            await this.media?.resumePlayback();
            await this.client.disconnect();
        } catch (error) {
            this.log(`disconnect failed: ${(error as Error).message}`);
        }
    }

    // ---- pause / resume ----------------------------------------------------

    async pause(): Promise<void> {
        if (!this.client || this.paused) return;
        this.paused = true;
        this.setButtons(true);
        this.client.enableMic(false);
        await this.media?.pausePlayback();
        this.client.sendClientMessage('pause');
        this.addTranscript('sys', 'Paused');
        this.log('pause sent');
    }

    async resume(): Promise<void> {
        if (!this.client || !this.paused) return;
        this.paused = false;
        this.setButtons(true);
        await this.media?.resumePlayback();
        this.client.sendClientMessage('resume');
        this.client.enableMic(true);
        this.addTranscript('sys', 'Resumed');
        this.log('resume sent');
    }

    // ---- report ------------------------------------------------------------

    private async showReport(): Promise<void> {
        // The server prints the report on disconnect; give it a moment then fetch it for display.
        await new Promise((r) => setTimeout(r, 800));
        try {
            const res = await fetch(`${SERVER}/report/latest`);
            const report = await res.json();
            if (report.empty) return;
            this.el.report.textContent = formatReport(report);
            this.el.reportCard.classList.remove('hidden');
        } catch (error) {
            this.log(`report fetch failed: ${(error as Error).message}`);
        }
    }
}

function formatReport(r: any): string {
    const lines: string[] = [];
    lines.push(`Duration ${r.duration_seconds}s · user turns ${r.turns.user} · bot turns ${r.turns.bot}`);
    const resp = r.latency?.user_to_bot_response;
    if (resp?.n) lines.push(`User → bot response: avg ${resp.avg}s  p95 ${resp.p95}s  (n=${resp.n})`);
    for (const [proc, s] of Object.entries<any>(r.latency?.ttfb ?? {})) {
        lines.push(`TTFB ${proc}: avg ${s.avg}s  max ${s.max}s  (n=${s.n})`);
    }
    for (const [model, t] of Object.entries<any>(r.llm_tokens ?? {})) {
        lines.push(`LLM ${model}: ${t.calls} calls · ${t.prompt} prompt + ${t.completion} completion = ${t.total} tokens`);
    }
    lines.push(`TTS: ${r.tts.calls} requests, ${r.tts.characters} chars · STT: ${r.stt.audio_seconds}s audio`);
    if (r.tool_calls && Object.keys(r.tool_calls).length) lines.push(`Tools: ${JSON.stringify(r.tool_calls)}`);
    if (r.events && Object.keys(r.events).length) lines.push(`Events: ${JSON.stringify(r.events)}`);
    if (r.controller) lines.push(`Lesson: ${JSON.stringify(r.controller)}`);
    lines.push(`Estimated cost: $${r.estimated_cost_usd}`);
    return lines.join('\n');
}

window.addEventListener('DOMContentLoaded', () => {
    new TutorApp();
});
