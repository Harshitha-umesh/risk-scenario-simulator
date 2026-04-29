# Simple in-memory cache (no Redis needed)

cache_store = {}

def get_cache(key):
    return cache_store.get(key)

def set_cache(key, value):
    cache_store[key] = value