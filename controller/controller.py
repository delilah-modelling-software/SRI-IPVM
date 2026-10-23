import pygame as pg

def ignore_events():
    pg.event.pump()

def wait(n_ms):
    pg.time.wait(n_ms)
    return

def delay(n_ms):
    pg.time.delay(n_ms)
    return

