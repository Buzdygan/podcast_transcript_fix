# Podcast transcript fixer

This small tool sends a **raw podcast transcript** (for example from automatic speech recognition) to Google’s **Gemini** AI. It returns cleaned-up text with better sentence breaks, paragraphs, spelling, and light editing—without changing what was actually said.

It is meant to run on your own computer. You only need **Python**, an **internet connection**, and a **free Google API key** for the Gemini API.

---

## What you need before you start

1. **A computer** with macOS, Windows, or Linux.
2. **Python 3.10 or newer** installed.  
   - Check by opening a terminal (on Mac: *Terminal*; on Windows: *PowerShell* or *Command Prompt*) and running:
     ```bash
     python3 --version
     ```
   - If that fails, try:
     ```bash
     python --version
     ```
   - If Python is missing or older than 3.10, install it from [python.org](https://www.python.org/downloads/) and use the installer option to “add Python to PATH” on Windows.
3. **A Google account** (a normal Gmail account is fine).
4. **This project folder** on your machine—either copied from a USB drive / cloud folder, or cloned with Git if you use it.

---

## Step 1: Open a terminal in the project folder

- **Mac / Linux:** In Finder, go into the `podcast_transcript_fix` folder, right‑click empty space, choose “Open in Terminal” if available, or open Terminal and `cd` into the folder, for example:
  ```bash
  cd ~/Downloads/podcast_transcript_fix
  ```
  (Adjust the path to wherever the folder actually is.)

- **Windows:** Open PowerShell, then:
  ```powershell
  cd C:\Users\YourName\Downloads\podcast_transcript_fix
  ```
  Again, use the real path to your copy of the project.

All commands below are run **from inside** this `podcast_transcript_fix` folder.

---

## Step 2: Create a virtual environment (recommended)

A virtual environment keeps this project’s Python packages separate from other programs.

**Mac / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

After this, your terminal prompt will often show `(.venv)`—that means the environment is active. **Use the same terminal** for the next steps.

If `python3` does not work on Windows, try `py -3 -m venv .venv` instead.

---

## Step 3: Install dependencies

With the virtual environment activated:

```bash
pip install -r requirements.txt
```

Wait until it finishes without errors. If `pip` is not found, try `python -m pip install -r requirements.txt`.

---

## Step 4: Get a Google Gemini API key (free tier)

The script talks to Google’s servers using an **API key**. You create the key once and store it in a file on your laptop.

1. Open **[Google AI Studio](https://aistudio.google.com/apikey)** in a browser and sign in with your Google account.
2. Click **Create API key** (or similar). You may be asked to pick a Google Cloud project; you can let it create a default project for you.
3. Copy the key string. **Treat it like a password**—do not post it publicly or email it in plain text.

Pricing note: Google offers **free usage** for Gemini models within certain limits; see Google’s current [Gemini API pricing](https://ai.google.dev/pricing) page for details. This project defaults to a model suitable for that kind of usage (`gemini-2.5-flash`).

---

## Step 5: Save the API key in a `.env` file

1. In the `podcast_transcript_fix` folder, find **`.env.example`**.
2. **Copy** it to a new file named **`.env`** (same folder, no `.example` in the name).
   - **Mac / Linux** terminal (from the project folder):
     ```bash
     cp .env.example .env
     ```
   - Or copy/paste in File Explorer / Finder.
3. Open **`.env`** in a text editor (Notepad, TextEdit, VS Code, etc.).
4. Replace `your_key_here` with your real API key, for example:
   ```env
   GOOGLE_API_KEY=AIzaSy...your_actual_key...
   ```
   You can use either variable name:
   - `GOOGLE_API_KEY=...` **or**
   - `GEMINI_API_KEY=...`

5. Save the file.

**Important:** The file must be named `.env` and live in the **same folder** as `fix_transcript.py`. Do **not** commit `.env` to public Git repositories; this project’s `.gitignore` is set to ignore it.

---

## Step 6: Prepare a transcript file

Put your podcast transcript in a **plain text** file, for example `episode.txt`, encoded as **UTF‑8** (normal for most editors).

You can keep the file anywhere; you will pass its path when you run the script.

---

## Step 7: Run the script

Make sure the virtual environment is still activated (`(.venv)` in the prompt). From the **project folder**:

```bash
python fix_transcript.py path/to/your/episode.txt
```

Example if the file is on your Desktop (Mac):

```bash
python fix_transcript.py ~/Desktop/episode.txt
```

### Where the result goes

By default the corrected text is written to:

```text
podcast_transcript_fix/output_data/<original-filename-without-extension>-fixed.txt
```

Example: input `episode.txt` → output `output_data/episode-fixed.txt`.

The script creates the `output_data` folder automatically if needed.

### Other useful options

| Option | Meaning |
|--------|--------|
| `-o some/other/file.txt` | Write the result to a specific file instead of the default. |
| `-o -` | Print the result to the terminal (stdout) instead of a file. |
| `--no-merge` | Skip the final “smooth the joins between chunks” step (faster, sometimes rougher on very long files). |
| `--chunk-size 12000` | Use larger chunks (default is 10000 characters). Rarely needed. |
| `--model <id>` | Set the Gemini model (default `gemini-2.5-flash`). Other common ids: `gemini-2.5-flash-lite`, `gemini-2.0-flash`, `gemini-1.5-flash`. |
| `--auto-fallback` | If the API keeps returning temporary overload errors (503), automatically try other models in a built-in order. |
| `--no-interactive` | Never ask questions in the terminal; on overload, print help and exit unless `--auto-fallback` can switch models. |

Full help:

```bash
python fix_transcript.py --help
```

---

## Troubleshooting

| Problem | What to try |
|--------|-------------|
| `No API key found` | Confirm `.env` is in the project folder next to `fix_transcript.py`, contains `GOOGLE_API_KEY=` or `GEMINI_API_KEY=` with no spaces around `=`, and you saved the file. |
| `python` / `python3` not found | Install Python from [python.org](https://www.python.org/downloads/) and restart the terminal. On Windows, try `py` instead of `python`. |
| `pip install` errors | Upgrade pip: `python -m pip install --upgrade pip`, then run `pip install -r requirements.txt` again. |
| Permission / API errors from Google | Check the key in AI Studio, billing/quota messages on Google’s side, and that the network allows HTTPS. |
| **`503 UNAVAILABLE` / “high demand”** | Google’s servers are busy or your free-tier quota for that model is momentarily used up. The script **retries** a few times, then can **switch models** (see below). You do **not** need to open the README—read the message printed in the terminal. |
| Very long episodes | The script already splits long text into pieces. If something fails, try `--no-merge` or a smaller `--chunk-size`. |

### Service busy (503) or overloaded

1. **Wait 2–15 minutes** and run the **same command** again. Demand spikes are often short.
2. **Pick another model**, for example `--model gemini-2.5-flash-lite` or `--model gemini-2.0-flash`.
3. Or run with **`--auto-fallback`** so the script tries other models automatically without prompts.
4. In scripts or CI where no one can answer a prompt, use **`--no-interactive`** together with **`--auto-fallback`** if you want automatic model switching.

If you run the tool **interactively** in a normal terminal, it may **ask which model to try next** and show exact model ids you can paste into `--model` later.

---

## Privacy

Running this tool sends your transcript text to **Google’s servers** for processing (same idea as using Gemini in a browser). Do not use it for content you are not allowed to share externally.

---

## Project contents (short)

| File / folder | Role |
|---------------|------|
| `fix_transcript.py` | Main script |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for your secret API key (copy to `.env`) |
| `output_data/` | Default folder for corrected `.txt` files (created when you run the script) |

If something in these steps does not match your computer, note your **operating system** and the **exact error message**—that makes it easier to fix.
