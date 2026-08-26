from langgraph.store.memory import InMemoryStore

# Create Memory Store
store = InMemoryStore()

# Create namespaces
namespace_1 = ('users', 'u1')
namespace_2 = ('users', 'u2')

# Add memories to namespaces
store.put(namespace_1, '1', {'data': 'User likes Pizza'})
store.put(namespace_1, '2', {'data': 'User Prefers dark mode'})

store.put(namespace_2, '1', {'data': 'User likes Pasta'})
store.put(namespace_2, '2', {'data': 'User prefers Grid style navigation'})


# Get memories from store
# print(store.get(namespace_1, '1'))
# print(store.get(namespace_2, '1'))

# Retriveing all memories
user1_memories = store.search(namespace_1)
user2_memories = store.search(namespace_2)

print('User 1 memores')
for memory in user1_memories:
    print(memory)

print('User 2 memores')
for memory in user2_memories:
    print(memory)

