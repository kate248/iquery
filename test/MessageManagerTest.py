from memory.MessageManager import  MessageManager

msg1 = MessageManager()
print(msg1.system_messages)
print(msg1.history_messages)
msg1.messages_append({"role": "user", "content": "你好，有什么可以帮你？"})
print(msg1.history_messages)
print("test")