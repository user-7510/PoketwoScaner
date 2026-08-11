import discord
import json

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

token = input('請輸入 Bot Token：')
channelId = int(input('請輸入頻道 ID：'))
messageId = int(input('請輸入訊息 ID：'))

@client.event
async def on_ready():
    channel = await client.fetch_channel(channelId)
    message = await channel.fetch_message(messageId)

    data = {
        'id': message.id,
        'content': message.content,
        'author': {
            'id': message.author.id,
            'name': message.author.name,
            'displayName': message.author.display_name,
            'bot': message.author.bot,
        },
        'channelId': message.channel.id,
        'guildId': message.guild.id if message.guild else None,
        'createdAt': message.created_at.isoformat(),
        'editedAt': message.edited_at.isoformat() if message.edited_at else None,
        'jumpUrl': message.jump_url,
        'pinned': message.pinned,
        'tts': message.tts,
        'type': str(message.type),
        'mentionEveryone': message.mention_everyone,
        'mentions': [u.id for u in message.mentions],
        'roleMentions': [r.id for r in message.role_mentions],
        'channelMentions': [c.id for c in message.channel_mentions],
        'attachments': [
            {'filename': a.filename, 'url': a.url, 'size': a.size, 'contentType': a.content_type}
            for a in message.attachments
        ],
        'embeds': [e.to_dict() for e in message.embeds],
        'reactions': [
            {'emoji': str(r.emoji), 'count': r.count} for r in message.reactions
        ],
        'stickers': [s.name for s in message.stickers],
        'flags': message.flags.value,
        'reference': message.reference.message_id if message.reference else None,
    }

    with open(f'{messageId}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f'已儲存至 {messageId}.json')
    await client.close()

client.run(token)
