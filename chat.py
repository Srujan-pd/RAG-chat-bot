def build_prompt(db, session_id, user_message, context_limit=10):
    """
    Build prompt with recent chat history for context.
    context_limit: Number of recent exchanges to include (default 10)
    """
    seven_days_ago = datetime.utcnow() - timedelta(days=7)

    chats = (
        db.query(Chat)
        .filter(
            Chat.session_id == session_id,
            Chat.created_at >= seven_days_ago
        )
        .order_by(Chat.created_at.desc())
        .limit(context_limit)
        .all()
    )

    # Reverse to get chronological order
    chats = list(reversed(chats))

    if chats:
        prompt = "You are a helpful support AI assistant. Previous conversation:\n\n"
        for chat in chats:
            prompt += f"User: {chat.question}\n"
            prompt += f"Assistant: {chat.answer}\n\n"
        prompt += f"User: {user_message}\nAssistant:"
    else:
        # First message - add greeting context
        if user_message.lower().strip() in ["hi", "hello", "hey", "hi there", "hello there", "hey there"]:
            prompt = """You are a friendly AI assistant for Primis Digital. Start with a warm greeting and introduce yourself briefly. Then ask how you can help.

User: {user_message}
Assistant:""".format(user_message=user_message)
        else:
            prompt = f"""You are a helpful AI assistant for Primis Digital. You provide information about Primis Digital's services, technologies, and projects.

User: {user_message}
Assistant:"""

    return prompt
