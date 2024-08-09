from OpenGL.GL import *


class Light:
    def __init__(self, light=GL_LIGHT0, ambient=[0.45, 0.45, 0.45, 1.0], diffuse=[1.0, 1.0, 1.0, 1.0],
                 position=[300.0, 800.0, 400.0, 1.0], specular=[0.8, 0.8, 0.8, 1.0]):
        self.light = light
        glLightfv(light, GL_AMBIENT, ambient)
        glLightfv(light, GL_DIFFUSE, diffuse)
        glLightfv(light, GL_POSITION, position)
        glLightfv(light, GL_SPECULAR, specular)

    def enable(self):
        glEnable(self.light)

    def disable(self):
        glDisable(self.light)

    def switch(self):
        if glGetBoolean(self.light) == GL_TRUE:
            glDisable(self.light)
        else:
            glEnable(self.light)
