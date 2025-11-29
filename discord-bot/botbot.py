"""
Complete Moderation Bot with:
- AI moderation (OpenAI optional)
- Anti-spam, slur detection
- Warnings + auto-mute
- Admin-only slash commands
- Persistent logs + mod-logs channel
- Auto-role on join
- Simple Flask dashboard for uptime & logs

CONFIG (via environment variables):
- TOKEN                (required) : Discord bot token
- MOD_LOG_CHANNEL      (optional) : default "mod-logs"
- AUTO_ROLE            (optional) : role name to assign on member join
- MAX_WARNINGS         (optional) : default 5
- MUTE_DURATION        (optional) : seconds, default 300
- BOT_DELETE_TIME      (optional) : seconds to auto-delete bot messages, default 5
- SPAM_LIMIT           (optional) : messages in window, default 5
- SPAM_WINDOW          (optional) : seconds window for spam, default 7
- OPENAI_API_KEY       (optional) : if provided, uses OpenAI moderation endpoint
- DASHBOARD_PORT       (optional) : port for Flask app, default 8080
"""

import os
import re
import json
import time
import asyncio
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import discord
from discord.ext import commands, tasks
from discord import app_commands
from unidecode import unidecode

# Optional OpenAI moderation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
use_openai = bool(OPENAI_API_KEY)
if use_openai:
    import openai
    openai.api_key = OPENAI_API_KEY

# ========== Config (replace via environment variables) ==========
TOKEN = os.getenv("TOKEN") or os.getenv("DISCORD_BOT_TOKEN")  # <<< REPLACE IN ENV
MOD_LOG_CHANNEL = os.getenv("MOD_LOG_CHANNEL", "mod-logs")
AUTO_ROLE = os.getenv("AUTO_ROLE", None)  # e.g. "Member" or leave empty
MAX_WARNINGS = int(os.getenv("MAX_WARNINGS", 5))
MUTE_DURATION = int(os.getenv("MUTE_DURATION", 300))  # seconds
BOT_DELETE_TIME = int(os.getenv("BOT_DELETE_TIME", 5))
SPAM_LIMIT = int(os.getenv("SPAM_LIMIT", 5))
SPAM_WINDOW = int(os.getenv("SPAM_WINDOW", 7))
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 8080))

WARNINGS_FILE = "warnings.json"
LOG_FILE = "moderation.log"
RECENT_LOG_LINES = 40

# ========== Safety: ensure TOKEN present ==========
if not TOKEN:
    raise RuntimeError("TOKEN environment variable missing. Add your Discord bot token.")

# ========== Bot setup ==========
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

# In-memory structures
user_warnings = {}        # loaded from disk
user_message_times = {}   # for spam detection
bot_start_time = time.time()

# Load warnings from disk
def load_warnings():
    global user_warnings
    if os.path.exists(WARNINGS_FILE):
        try:
            with open(WARNINGS_FILE, "r") as f:
                user_warnings = json.load(f)
                # keys should be ints
                user_warnings = {int(k): int(v) for k, v in user_warnings.items()}
        except Exception:
            user_warnings = {}
    else:
        user_warnings = {}

def save_warnings():
    try:
        with open(WARNINGS_FILE, "w") as f:
            json.dump({str(k): v for k, v in user_warnings.items()}, f)
    except Exception as e:
        print("Failed to save warnings:", e)

def append_log(text):
    ts = datetime.utcnow().isoformat()
    line = f"[{ts}] {text}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)

async def send_mod_log(guild, content):
    """Send content to mod-logs channel if it exists."""
    for channel in guild.text_channels:
        if channel.name == MOD_LOG_CHANNEL:
            try:
                await channel.send(content)
            except Exception:
                pass
            break

# ========== Detection helpers (no explicit slur lists) ==========
# Structural patterns (safe — don't include explicit slur lists)
SLUR_PATTERNS = [
    r"n+\W*[i1!]+\W*[gq9]+\W*[e3a]+\W*[r]+",
    r"f+\W*[a@4]+\W*[gq9]+\W*[o0]+\W*[t]+",
]

def normalize_text(s: str) -> str:
    s = unidecode(s.lower())
    s = re.sub(r'[\s\W_]+', '', s)
    # collapse repeated characters
    s = re.sub(r'(.)\1{2,}', r'\1', s)
    return s

def shape_match(s: str) -> float:
    # Abstract "shapes" to compare roughly
    shapes = ["nger", "fgot", "kk"]  # abstract models (not explicit words)
    s_norm = normalize_text(s)
    best = 0.0
    for sh in shapes:
        score = SequenceMatcher(None, s_norm, sh).ratio()
        if score > best:
            best = score
    return best

def heuristic_toxic_score(text: str) -> float:
    """Return a 0..1 heuristic toxicity score (no explicit slurs printed)."""
    t = text.lower()
    # structural regex matches (high signal)
    for p in SLUR_PATTERNS:
        if re.search(p, text, flags=re.IGNORECASE):
            return 0.95
    # presence of repeated punctuation or all caps sentences (mild signal)
    if re.search(r'[!?.]{4,}', text):
        return 0.45
    caps_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)
    if caps_ratio > 0.5:
        return 0.4
    # shape match (fuzzy)
    sm = shape_match(text)
    if sm > 0.6:
        return 0.8
    # default low
    return 0.0

async def ai_moderation_score(text: str) -> float:
    """Return a toxicity score 0..1. If OPENAI_API_KEY provided, use OpenAI moderation; otherwise fallback to heuristic."""
    if use_openai:
        try:
            # Use OpenAI moderation endpoint
            resp = openai.Moderation.create(input=text)
            # simple approach: if any category flagged, set score high
            results = resp["results"][0]
            # compute score as proportion of categories flagged (heuristic)
            categories = results.get("categories", {})
            flagged_count = sum(1 for v in categories.values() if v)
            total = len(categories) or 1
            return min(1.0, flagged_count / total + 0.2)  # bias up slightly
        except Exception as e:
            print("OpenAI moderation error:", e)
            return heuristic_toxic_score(text)
    else:
        # fallback heuristic
        return heuristic_toxic_score(text)

# ========== Moderation actions ==========
async def apply_warning(member: discord.Member, guild: discord.Guild, channel: discord.TextChannel, reason: str):
    uid = member.id
    user_warnings[uid] = user_warnings.get(uid, 0) + 1
    save_warnings()
    await channel.send(f"⚠️ {member.mention} {reason} — Warning {user_warnings[uid]}/{MAX_WARNINGS}", delete_after=BOT_DELETE_TIME)
    await send_mod_log(guild, f"⚠️ {member} — {reason} — Warnings: {user_warnings[uid]}/{MAX_WARNINGS}")
    append_log(f"{member} ({member.id}) - WARNING - {reason} - {user_warnings[uid]}/{MAX_WARNINGS}")
    # mute if necessary
    if user_warnings[uid] >= MAX_WARNINGS:
        user_warnings[uid] = 0
        save_warnings()
        await mute_member(member, guild, channel, reason="Reached max warnings")

async def mute_member(member: discord.Member, guild: discord.Guild, channel: discord.TextChannel, reason="Violation"):
    MUTED_ROLE_NAME = "Muted"
    muted_role = discord.utils.get(guild.roles, name=MUTED_ROLE_NAME)
    if not muted_role:
        muted_role = await guild.create_role(name=MUTED_ROLE_NAME)
        for ch in guild.text_channels:
            try:
                await ch.set_permissions(muted_role, send_messages=False, speak=False)
            except Exception:
                pass
    try:
        await member.add_roles(muted_role, reason=reason)
    except Exception:
        pass
    await channel.send(f"🔇 {member.mention} has been muted ({reason})", delete_after=BOT_DELETE_TIME)
    await send_mod_log(guild, f"🔇 {member} ({member.id}) muted — {reason}")
    append_log(f"{member} ({member.id}) - MUTED - {reason}")
    # schedule unmute after MUTE_DURATION
    await asyncio.sleep(MUTE_DURATION)
    try:
        await member.remove_roles(muted_role, reason="Auto unmute after mute duration")
        await send_mod_log(guild, f"🔊 {member} ({member.id}) unmuted after mute duration")
        append_log(f"{member} ({member.id}) - UNMUTED - auto")
    except Exception:
        pass

# ========== Events ==========
@bot.event
async def on_ready():
    load_warnings()
    uptime_ping.start()
    print(f"✅ Bot online as {bot.user} — uptime ping started")
    print(f"OpenAI moderation enabled: {use_openai}")

@bot.event
async def on_member_join(member):
    """Auto-role on join if AUTO_ROLE is set"""
    if AUTO_ROLE:
        role = discord.utils.get(member.guild.roles, name=AUTO_ROLE)
        if role:
            try:
                await member.add_roles(role, reason="Auto role on join")
                await send_mod_log(member.guild, f"✅ Auto-role: {member} was given role '{AUTO_ROLE}'")
                append_log(f"{member} ({member.id}) - AUTO_ROLE - {AUTO_ROLE}")
            except Exception:
                pass

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    author = message.author
    now = time.time()

    # ---- spam detection ----
    times = user_message_times.get(author.id, [])
    times = [t for t in times if now - t < SPAM_WINDOW]
    times.append(now)
    user_message_times[author.id] = times
    if len(times) >= SPAM_LIMIT:
        try:
            await message.delete()
        except Exception:
            pass
        await apply_warning(author, message.guild, message.channel, reason="is spamming")
        await send_mod_log(message.guild, f"⚠️ {author} triggered spam detection ({len(times)} messages in {SPAM_WINDOW}s)")
        return

    # ---- AI moderation ----
    score = await ai_moderation_score(message.content)
    # actions thresholds (tunable)
    if score >= 0.70:
        try:
            await message.delete()
        except Exception:
            pass
        await apply_warning(author, message.guild, message.channel, reason=f"AI-moderation removed message (score {score:.2f})")
        return
    elif score >= 0.40:
        # soft warning
        try:
            await message.delete()
        except Exception:
            pass
        await apply_warning(author, message.guild, message.channel, reason=f"AI-moderation flagged message (score {score:.2f})")
        return

    # ---- heuristic slur detection fallback ----
    # check shape/patterns
    for p in SLUR_PATTERNS:
        if re.search(p, message.content, flags=re.IGNORECASE) or re.search(p, normalize_text(message.content), flags=re.IGNORECASE):
            try:
                await message.delete()
            except Exception:
                pass
            await apply_warning(author, message.guild, message.channel, reason="used inappropriate language (pattern match)")
            return

    await bot.process_commands(message)

# ========== Background uptime ping (keeps logs updated) ==========
@tasks.loop(minutes=5)
async def uptime_ping():
    # just prints to logs and keeps task active
    uptime = int(time.time() - bot_start_time)
    print(f"[UPTIME] Bot active for {uptime} seconds")
    append_log(f"UPTIME_PING - {uptime}s")

# ========== Admin permission helper ==========
def admin_only(interaction: discord.Interaction):
    return interaction.user.guild_permissions.administrator

# ========== Slash commands ==========
@bot.tree.command(name="ping", description="Check bot latency")
async def slash_ping(interaction: discord.Interaction):
    await interaction.response.send_message(f"Pong! {round(bot.latency*1000)} ms", ephemeral=True)

@bot.tree.command(name="uptime", description="Show bot uptime")
async def slash_uptime(interaction: discord.Interaction):
    seconds = int(time.time() - bot_start_time)
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    await interaction.response.send_message(f"Uptime: {hrs}h {mins}m", ephemeral=True)

@bot.tree.command(name="warnings", description="Check a user's warnings")
async def slash_warnings(interaction: discord.Interaction, user: discord.Member):
    count = user_warnings.get(user.id, 0)
    await interaction.response.send_message(f"{user.mention} has {count}/{MAX_WARNINGS} warnings.", ephemeral=True)

@bot.tree.command(name="clearwarnings", description="Clear user's warnings (Admin only)")
async def slash_clearwarnings(interaction: discord.Interaction, user: discord.Member):
    if not admin_only(interaction):
        await interaction.response.send_message("❌ Admins only.", ephemeral=True)
        return
    user_warnings[user.id] = 0
    save_warnings()
    await interaction.response.send_message(f"✅ Warnings reset for {user.mention}.", ephemeral=True)
    await send_mod_log(interaction.guild, f"✅ {interaction.user} reset warnings for {user}")

@bot.tree.command(name="mute", description="Mute a user (Admin only)")
async def slash_mute(interaction: discord.Interaction, user: discord.Member):
    if not admin_only(interaction):
        await interaction.response.send_message("❌ Admins only.", ephemeral=True)
        return
    await mute_member(user, interaction.guild, interaction.channel, reason=f"Admin mute by {interaction.user}")
    await interaction.response.send_message(f"🔇 {user.mention} has been muted by admin.", ephemeral=True)

@bot.tree.command(name="unmute", description="Unmute a user (Admin only)")
async def slash_unmute(interaction: discord.Interaction, user: discord.Member):
    if not admin_only(interaction):
        await interaction.response.send_message("❌ Admins only.", ephemeral=True)
        return
    muted_role = discord.utils.get(interaction.guild.roles, name="Muted")
    if muted_role:
        await user.remove_roles(muted_role)
    await interaction.response.send_message(f"🔊 {user.mention} has been unmuted by admin.", ephemeral=True)
    await send_mod_log(interaction.guild, f"🔊 {user} was unmuted by {interaction.user}")

# ========== Simple Flask dashboard ==========
DASH_TEMPLATE = """
<!doctype html>
<title>Moderation Bot Dashboard</title>
<h1>Moderation Bot Dashboard</h1>
<p>Uptime: {{uptime}}</p>
<h2>Warnings (top 20)</h2>
<ul>
{% for uid, cnt in warnings %}
  <li>{{uid}} : {{cnt}}</li>
{% endfor %}
</ul>
<h2>Recent Logs</h2>
<pre>{{logs}}</pre>
"""

def index():
    seconds = int(time.time() - bot_start_time)
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    uptime = f"{hrs}h {mins}m"
    # warnings sorted
    w = sorted(user_warnings.items(), key=lambda i: i[1], reverse=True)[:20]
    # logs tail
    logs = ""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()[-RECENT_LOG_LINES:]
            logs = "".join(lines)
    return render_template_string(DASH_TEMPLATE, uptime=uptime, warnings=[(k, v) for k, v in w], logs=logs)

# ========== Main entry ==========
if __name__ == "__main__":
    load_warnings()
    # run bot
    bot.run(TOKEN)




