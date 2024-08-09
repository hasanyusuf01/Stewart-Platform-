from OpenGL.GL import *
import numpy as np
import stl

class Drawable:
    def draw(self):
        pass


class Cube(Drawable):
    def __init__(self, size=70.):
        if isinstance(size, (list, tuple, np.ndarray)):
            self.half_x, self.half_y, self.half_z = size[0], size[1], size[2]
        else:
            self.half_x = self.half_y = self.half_z = size

        self.model = glGenLists(1)
        glNewList(self.model, GL_COMPILE)
        glBegin(GL_QUADS)

        glNormal3f(0.0, 0.0, -1.0)
        glVertex3f(-self.half_x, -self.half_y, -self.half_z)
        glVertex3f(-self.half_x, self.half_y, -self.half_z)
        glVertex3f(self.half_x, self.half_y, -self.half_z)
        glVertex3f(self.half_x, -self.half_y, -self.half_z)

        glNormal3f(0.0, -1.0, 0.0)
        glVertex3f(-self.half_x, -self.half_y, -self.half_z)
        glVertex3f(self.half_x, -self.half_y, -self.half_z)
        glVertex3f(self.half_x, -self.half_y, self.half_z)
        glVertex3f(-self.half_x, -self.half_y, self.half_z)

        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(-self.half_x, -self.half_y, -self.half_z)
        glVertex3f(-self.half_x, -self.half_y, self.half_z)
        glVertex3f(-self.half_x, self.half_y, self.half_z)
        glVertex3f(-self.half_x, self.half_y, -self.half_z)

        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(-self.half_x, -self.half_y, self.half_z)
        glVertex3f(self.half_x, -self.half_y, self.half_z)
        glVertex3f(self.half_x, self.half_y, self.half_z)
        glVertex3f(-self.half_x, self.half_y, self.half_z)

        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(-self.half_x, self.half_y, -self.half_z)
        glVertex3f(-self.half_x, self.half_y, self.half_z)
        glVertex3f(self.half_x, self.half_y, self.half_z)
        glVertex3f(self.half_x, self.half_y, -self.half_z)

        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(self.half_x, -self.half_y, -self.half_z)
        glVertex3f(self.half_x, self.half_y, -self.half_z)
        glVertex3f(self.half_x, self.half_y, self.half_z)
        glVertex3f(self.half_x, -self.half_y, self.half_z)

        glEnd()
        glEndList()

    def draw(self):
        glCallList(self.model)


class Disk(Drawable):
    def __init__(self, radius=25., slices=32):
        self.radius, self.slices = radius, slices

        self.vertices = np.array([[np.cos(angle)*self.radius, np.sin(angle)*self.radius, 0]
                                  for angle in [i * 2 * np.pi / self.slices for i in range(slices)]])

    def draw(self):
        glBegin(GL_TRIANGLES)
        for i in range(self.slices):
            j = i+1 if i < self.slices-1 else 0
            glVertex3f(*self.vertices[j])
            glVertex3f(0, 0, 0)
            glVertex3f(*self.vertices[i])
        glEnd()


class Cylinder(Drawable):
    def __init__(self, height=25., radius=25., slices=32):
        self.radius, self.slices, self.height = radius, slices, height
        _angles = [i * 2 * np.pi / self.slices for i in range(slices)]
        self.base_vertices = np.array([[np.cos(angle)*self.radius, np.sin(angle)*self.radius, 0]
                                       for angle in _angles])
        self.top_vertices = np.array([[np.cos(angle) * self.radius, np.sin(angle) * self.radius, 2*height]
                                      for angle in _angles])

    def draw(self):
        glBegin(GL_QUADS)
        for i in range(self.slices):
            j = i + 1 if i < self.slices - 1 else 0
            glVertex3f(*self.base_vertices[i])
            glVertex3f(*self.base_vertices[j])
            glVertex3f(*self.top_vertices[j])
            glVertex3f(*self.top_vertices[i])
        glEnd()

        glBegin(GL_TRIANGLES)
        for i in range(self.slices):
            j = i + 1 if i < self.slices - 1 else 0
            glVertex3f(*self.base_vertices[i])
            glVertex3f(0, 0, 0)
            glVertex3f(*self.base_vertices[j])
        glEnd()

        glBegin(GL_TRIANGLES)
        for i in range(self.slices):
            j = i + 1 if i < self.slices - 1 else 0
            glVertex3f(*self.top_vertices[i])
            glVertex3f(0, 0, 2*self.height)
            glVertex3f(*self.top_vertices[j])
        glEnd()


class Cone(Drawable):
    def __init__(self, height=25, radius=25, slices=32):
        self.radius, self.height, self.slices = radius, height, slices

        self.base_vertices = np.array([[np.cos(angle)*self.radius, np.sin(angle)*self.radius, 0]
                                       for angle in [i * 2 * np.pi / self.slices for i in range(slices)]])
        self.top_point = np.array([0, 0, 2*height])

    def draw(self):
        glBegin(GL_TRIANGLES)
        for i in range(self.slices):
            j = i+1 if i < self.slices-1 else 0
            glVertex3f(*self.base_vertices[i])
            glVertex3f(*self.top_point)
            glVertex3f(*self.base_vertices[j])
        glEnd()


class CoordinateSystem(Drawable):
    def __init__(self, length=100, cone_height=10, cone_radius=10, on_top=True):
        self.length, self.cone_height, self.cone_radius, self.on_top = length, cone_height, cone_radius, on_top
        self.cone = Cone(self.cone_height, self.cone_radius)

        self.model = glGenLists(1)
        glNewList(self.model, GL_COMPILE)
        glPushMatrix()
        glLineWidth(0.03 * self.length)

        glPushMatrix()
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(self.length, 0, 0)
        glEnd()
        glTranslatef(self.length - self.cone_height, 0, 0)
        glRotatef(90, 0, 1, 0)
        self.cone.draw()
        glPopMatrix()

        glPushMatrix()
        glBegin(GL_LINES)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, self.length, 0)
        glEnd()
        glTranslatef(0, self.length - self.cone_height, 0)
        glRotatef(-90, 1, 0, 0)
        self.cone.draw()
        glPopMatrix()

        glPushMatrix()
        glBegin(GL_LINES)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, self.length)
        glEnd()
        glTranslatef(0, 0, self.length - self.cone_height)
        self.cone.draw()
        glPopMatrix()

        glPopMatrix()
        glEndList()

    def draw(self):
        pre_state = glGetBoolean(GL_LIGHTING)
        glDisable(GL_LIGHTING)

        if self.on_top:
            glDepthRange(0, 0.01)

        glCallList(self.model)

        if self.on_top:
            glDepthRange(0.01, 1.0)

        if pre_state == GL_TRUE:
            glEnable(GL_LIGHTING)


class Sphere(Drawable):
    def __init__(self, radius=25, slices=32):
        self.radius, self.latitude, self.longitude = radius, slices, slices

        self.model = glGenLists(1)
        glNewList(self.model, GL_COMPILE)
        for i in range(self.latitude+1):
            lat0 = np.pi * (-0.5 + (i - 1) / self.latitude)
            z0 = np.sin(lat0) * self.radius
            zr0 = np.cos(lat0) * self.radius

            lat1 = np.pi * (-0.5 + i / self.latitude)
            z1 = np.sin(lat1) * self.radius
            zr1 = np.cos(lat1) * self.radius

            glBegin(GL_QUAD_STRIP)
            for j in range(self.longitude+1):
                lng = 2 * np.pi * (j - 1) / self.longitude
                x = np.cos(lng)
                y = np.sin(lng)

                glNormal3d(x * zr0, y * zr0, z0)
                glVertex3d(x * zr0, y * zr0, z0)
                glNormal3d(x * zr1, y * zr1, z1)
                glVertex3d(x * zr1, y * zr1, z1)
            glEnd()
        glEndList()

    def draw(self):
        glCallList(self.model)


class STLModel(Drawable):
    def __init__(self, file_name='models/test/monkey.stl', calculate_normals=False):
        self.mesh = stl.mesh.Mesh.from_file(file_name, calculate_normals)

        self.model = glGenLists(1)
        glNewList(self.model, GL_COMPILE)
        glBegin(GL_TRIANGLES)
        for v, n in zip(self.mesh.vectors, self.mesh.normals):
            glNormal3f(*n)
            glVertex3f(*v[0])
            glVertex3f(*v[1])
            glVertex3f(*v[2])
        glEnd()
        glEndList()

    def draw(self):
        glCallList(self.model)

    def volume(self):
        return self.mesh.get_mass_properties()[0]

    def cog(self):
        return self.mesh.get_mass_properties()[1]

    def inertia(self):
        return self.mesh.get_mass_properties()[2]


class Mario(STLModel):
    def __init__(self):
        super().__init__('models/test/mario.stl')


class Tiles(Drawable):
    def __init__(self, row, col, size, alpha=0.2, color_diff=0.3):
        self.gen_id = glGenLists(1)
        glNewList(self.gen_id, GL_COMPILE)
        glPushMatrix()
        glTranslatef(-(size * row) / 2, -(size * col) / 2, 0)
        #lights_enabled = glGetBoolean(GL_LIGHTING)
        #glDisable(GL_LIGHTING)

        for i in range(row):
            for j in range(col):
                if (i + j) % 2 == 0:
                    glColor4f(0.5-color_diff, 0.5-color_diff, 0.5-color_diff, alpha)
                else:
                    glColor4f(0.5+color_diff, 0.5+color_diff, 0.5+color_diff, alpha)
                glRecti(i * size, j * size, (i + 1) * size, (j + 1) * size)
        #if lights_enabled:
        #    glEnable(GL_LIGHTING)
        glPopMatrix()
        glEndList()

    def draw(self):
        glPushMatrix()
        glTranslatef(0, 0, -0.1)
        glCallList(self.gen_id)
        glPopMatrix()


class Gradient(Drawable):
    def __init__(self, top, bottom):
        self.gen_id = glGenLists(1)
        glNewList(self.gen_id, GL_COMPILE)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        glLoadIdentity()
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        glBegin(GL_QUADS)
        glColor3f(top[0] / 255., top[1] / 255., top[2] / 255.)
        glVertex3f(-1.0, -1.0, -1.0)
        glVertex3f(1.0, -1.0, -1.0)

        glColor3f(bottom[0] / 255., bottom[1] / 255., bottom[2] / 255.)
        glVertex3f(1.0, 1.0, -1.0)
        glVertex3f(-1.0, 1.0, -1.0)
        glEnd()

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);
        glPopMatrix();

        glMatrixMode(GL_MODELVIEW);
        glPopMatrix()
        glEndList()

    def draw(self):
        glCallList(self.gen_id)