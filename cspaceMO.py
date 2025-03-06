import math
from math import sin, cos, acos, atan2, sqrt
from queue import PriorityQueue
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ------------------------------------------------------------------------
# Global variables for tie-breaking and angle range
# ------------------------------------------------------------------------
tie_breaker = 0
qMin = -math.pi    # minimum angle: -pi
qMax =  math.pi    # maximum angle: +pi
gridSteps = 60     # number of discretization steps per joint

# ------------------------------------------------------------------------
# A simple 2D Vector class
# ------------------------------------------------------------------------
class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def copy(self):
        return Vector(self.x, self.y)

# ------------------------------------------------------------------------
# Obstacle class
# ------------------------------------------------------------------------
class Obstacle:
    def __init__(self, center, r):
        self.center = center  # center as a Vector
        self.r = r            # radius

# ------------------------------------------------------------------------
# Helper functions for distances and collision checks
# ------------------------------------------------------------------------
def distSq(a, b):
    return (a.x - b.x)**2 + (a.y - b.y)**2

def distPoints(a, b):
    return math.sqrt(distSq(a, b))

def distToSegment(p, v, w):
    """Returns the distance from point p to the segment defined by v and w."""
    l2 = distSq(v, w)
    if l2 == 0:
        return distPoints(p, v)
    t = ((p.x - v.x)*(w.x - v.x) + (p.y - v.y)*(w.y - v.y)) / l2
    t = max(0, min(1, t))
    proj = Vector(v.x + t*(w.x - v.x), v.y + t*(w.y - v.y))
    return distPoints(p, proj)

def checkCollisionLine(obstacle, a, b):
    """Returns True if the segment from a to b collides with the given obstacle."""
    d = distToSegment(obstacle.center, a, b)
    return d <= obstacle.r

# ------------------------------------------------------------------------
# Robot class with its kinematics
# ------------------------------------------------------------------------
class Robot:
    def __init__(self, bx, by, L1, L2, L3):
        self.base = Vector(bx, by)  # base position
        self.L1 = L1                # length of link1
        self.L2 = L2                # length of link2
        self.L3 = L3                # length of link3
        # Initial joint angles
        self.q1 = math.pi/4
        self.q2 = math.pi/4
        self.q3 = -(self.q1 + self.q2)

    def getPoints(self):
        """Computes and returns the joint positions (p0, p1, p2, p3)."""
        p0 = self.base.copy()
        p1 = Vector(p0.x + self.L1 * cos(self.q1),
                    p0.y + self.L1 * sin(self.q1))
        p2 = Vector(p1.x + self.L2 * cos(self.q1 + self.q2),
                    p1.y + self.L2 * sin(self.q1 + self.q2))
        p3 = Vector(p2.x + self.L3 * cos(self.q1 + self.q2 + self.q3),
                    p2.y + self.L3 * sin(self.q1 + self.q2 + self.q3))
        return (p0, p1, p2, p3)

# ------------------------------------------------------------------------
# Check if a given configuration collides with ANY of the obstacles
# ------------------------------------------------------------------------
def configCollidesAny(robot, obstacles, q1, q2, q3):
    """
    Returns True if the given configuration (q1,q2,q3) causes the robot
    to collide with any obstacle in the list 'obstacles'.
    """
    # Save current angles
    oldQ1, oldQ2, oldQ3 = robot.q1, robot.q2, robot.q3
    # Set to new configuration
    robot.q1, robot.q2, robot.q3 = q1, q2, q3
    p0, p1, p2, p3 = robot.getPoints()

    collision = False
    for obs in obstacles:
        # If any link collides with this obstacle, we mark collision = True
        if (checkCollisionLine(obs, p0, p1) or
            checkCollisionLine(obs, p1, p2) or
            checkCollisionLine(obs, p2, p3)):
            collision = True
            break

    # Restore original configuration
    robot.q1, robot.q2, robot.q3 = oldQ1, oldQ2, oldQ3
    return collision

# ------------------------------------------------------------------------
# Discretization: map index to angle and vice versa
# ------------------------------------------------------------------------
def indexToAngle(idx, qMin, qMax, gridSteps):
    return qMin + (qMax - qMin) * idx / (gridSteps - 1)

def angleToIndex(q, qMin, qMax, gridSteps):
    if q < qMin:
        q = qMin
    if q > qMax:
        q = qMax
    val = (q - qMin) / (qMax - qMin) * (gridSteps - 1)
    return int(round(val))

# ------------------------------------------------------------------------
# Generate the C-space grid (3D boolean grid) for MULTIPLE obstacles
# ------------------------------------------------------------------------
def generateCSpaceGrid(robot, obstacles, gridSteps, qMin, qMax):
    """
    cspaceGrid[i][j][k] = True if config (i,j,k) is in collision with ANY obstacle.
    """
    cspaceGrid = [[[False for _ in range(gridSteps)]
                           for _ in range(gridSteps)]
                           for _ in range(gridSteps)]
    for i in range(gridSteps):
        for j in range(gridSteps):
            for k in range(gridSteps):
                a1 = indexToAngle(i, qMin, qMax, gridSteps)
                a2 = indexToAngle(j, qMin, qMax, gridSteps)
                a3 = indexToAngle(k, qMin, qMax, gridSteps)
                cspaceGrid[i][j][k] = configCollidesAny(robot, obstacles, a1, a2, a3)
    return cspaceGrid

def buildCollisionListForViz(cspaceGrid, gridSteps, qMin, qMax):
    """
    Return a list of (q1, q2, q3) for all cells that are in collision (True).
    """
    collisions = []
    for i in range(gridSteps):
        for j in range(gridSteps):
            for k in range(gridSteps):
                if cspaceGrid[i][j][k]:
                    a1 = indexToAngle(i, qMin, qMax, gridSteps)
                    a2 = indexToAngle(j, qMin, qMax, gridSteps)
                    a3 = indexToAngle(k, qMin, qMax, gridSteps)
                    collisions.append((a1, a2, a3))
    return collisions

# ------------------------------------------------------------------------
# A* for planning a path in C-space
# ------------------------------------------------------------------------
class Node3D:
    def __init__(self, i, j, k):
        self.i = i
        self.j = j
        self.k = k
        self.g = 0.0
        self.h = 0.0
        self.f = 0.0
        self.parent = None

def heuristic(i, j, k, gi, gj, gk):
    return math.dist((i, j, k), (gi, gj, gk))

def findPathAStar(cspaceGrid, startQ, goalQ, qMin, qMax, gridSteps):
    global tie_breaker

    si = angleToIndex(startQ[0], qMin, qMax, gridSteps)
    sj = angleToIndex(startQ[1], qMin, qMax, gridSteps)
    sk = angleToIndex(startQ[2], qMin, qMax, gridSteps)

    gi = angleToIndex(goalQ[0], qMin, qMax, gridSteps)
    gj = angleToIndex(goalQ[1], qMin, qMax, gridSteps)
    gk = angleToIndex(goalQ[2], qMin, qMax, gridSteps)

    # If start or goal are in collision, return None
    if cspaceGrid[si][sj][sk] or cspaceGrid[gi][gj][gk]:
        return None

    openSet = PriorityQueue()
    closed = [[[False for _ in range(gridSteps)]
                     for _ in range(gridSteps)]
                     for _ in range(gridSteps)]
    allNodes = [[[None for _ in range(gridSteps)]
                      for _ in range(gridSteps)]
                      for _ in range(gridSteps)]

    startNode = Node3D(si, sj, sk)
    startNode.g = 0.0
    startNode.h = heuristic(si, sj, sk, gi, gj, gk)
    startNode.f = startNode.g + startNode.h

    tie_breaker += 1
    openSet.put((startNode.f, tie_breaker, startNode))
    allNodes[si][sj][sk] = startNode

    goalNode = None

    # 26-connected neighborhood
    di = [ -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1 ]
    dj = [ -1, -1, -1,  0,  0,  1,  1,  1, -1,  0,  0,  1, -1,  0,  0,  1, -1, -1, -1,  0,  0,  1,  1,  1 ]
    dk = [ -1,  0,  1, -1,  1, -1,  0,  1, -1, -1,  1,  1,  0,  0,  0,  0, -1,  0,  1, -1,  1, -1,  0,  1 ]

    while not openSet.empty():
        _, _, current = openSet.get()
        if current.i == gi and current.j == gj and current.k == gk:
            goalNode = current
            break
        closed[current.i][current.j][current.k] = True

        for n in range(len(di)):
            ni = current.i + di[n]
            nj = current.j + dj[n]
            nk = current.k + dk[n]
            if ni < 0 or ni >= gridSteps or nj < 0 or nj >= gridSteps or nk < 0 or nk >= gridSteps:
                continue
            if cspaceGrid[ni][nj][nk]:
                continue
            if closed[ni][nj][nk]:
                continue

            cost = math.dist((current.i, current.j, current.k), (ni, nj, nk))
            tentative_g = current.g + cost

            neighbor = allNodes[ni][nj][nk]
            if neighbor is None:
                neighbor = Node3D(ni, nj, nk)
                allNodes[ni][nj][nk] = neighbor

            if tentative_g < neighbor.g or neighbor.g == 0:
                neighbor.g = tentative_g
                neighbor.h = heuristic(ni, nj, nk, gi, gj, gk)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current

                tie_breaker += 1
                openSet.put((neighbor.f, tie_breaker, neighbor))

    if goalNode is None:
        return None

    path = []
    n = goalNode
    while n is not None:
        qq1 = indexToAngle(n.i, qMin, qMax, gridSteps)
        qq2 = indexToAngle(n.j, qMin, qMax, gridSteps)
        qq3 = indexToAngle(n.k, qMin, qMax, gridSteps)
        path.insert(0, (qq1, qq2, qq3))
        n = n.parent
    return path

# ------------------------------------------------------------------------
# Inverse Kinematics (IK)
# ------------------------------------------------------------------------
def computeIK(robot, target):
    """
    Compute inverse kinematics such that the end effector points 'right'.
    """
    wx = target.x - robot.L3
    wy = target.y

    dx = wx - robot.base.x
    dy = wy - robot.base.y
    dBase = math.sqrt(dx*dx + dy*dy)

    qs = [0.0, 0.0, 0.0]

    if dBase > (robot.L1 + robot.L2):
        theta = math.atan2(dy, dx)
        qs[0] = theta
        qs[1] = 0.0
    else:
        cosAngle2 = ((dx*dx + dy*dy) - robot.L1**2 - robot.L2**2) / (2 * robot.L1 * robot.L2)
        cosAngle2 = max(-1, min(1, cosAngle2))
        angle2 = math.acos(cosAngle2)
        qs[1] = angle2

        k1 = robot.L1 + robot.L2 * math.cos(angle2)
        k2 = robot.L2 * math.sin(angle2)
        qs[0] = math.atan2(dy, dx) - math.atan2(k2, k1)
    qs[2] = - (qs[0] + qs[1])
    return qs

# ------------------------------------------------------------------------
# Interpolate the path for smoother motion
# ------------------------------------------------------------------------
def interpolatePath(originalPath, stepsPerSegment=10):
    """
    Linearly interpolates between each pair of configurations in the path.
    """
    if len(originalPath) < 2:
        return originalPath

    newPath = []
    for i in range(len(originalPath) - 1):
        q1_curr, q2_curr, q3_curr = originalPath[i]
        q1_next, q2_next, q3_next = originalPath[i+1]
        for step in range(stepsPerSegment):
            alpha = step / float(stepsPerSegment)
            q1_int = q1_curr * (1 - alpha) + q1_next * alpha
            q2_int = q2_curr * (1 - alpha) + q2_next * alpha
            q3_int = q3_curr * (1 - alpha) + q3_next * alpha
            newPath.append((q1_int, q2_int, q3_int))
    newPath.append(originalPath[-1])
    return newPath

# ------------------------------------------------------------------------
# MAIN function with animation (using FuncAnimation)
# ------------------------------------------------------------------------
def main():
    width = 1000
    height = 600

    # 1) Create the robot
    robot = Robot(width * 3/4, height / 2, 80, 80, 60)

    # 2) Create MULTIPLE obstacles
    # Here we define three obstacles at different positions with different radii
    obstaculo1 = Obstacle(Vector(robot.base.x + 100, robot.base.y + 40), 20)
    obstaculo2 = Obstacle(Vector(robot.base.x + 50, robot.base.y - 80), 15)
    obstaculo3 = Obstacle(Vector(robot.base.x + 160, robot.base.y - 20), 18)

    # Put them all in a list
    obstaculos = [obstaculo1, obstaculo2, obstaculo3]

    # 3) Define targets
    targets = [
        Vector(robot.base.x + 170, robot.base.y - 120),
        Vector(robot.base.x + 20,  robot.base.y + 100)
    ]
    desiredIKs = [computeIK(robot, t) for t in targets]

    # 4) Generate C-space grid (each joint in [-pi, +pi]) considering ALL obstacles
    print(f"Generating C-space grid with gridSteps={gridSteps}, angles in [{qMin}, {qMax}]...")
    cspaceGrid = generateCSpaceGrid(robot, obstaculos, gridSteps, qMin, qMax)
    print("C-space grid done.")

    # Build a list of collisions for visualization
    cSpaceCollisions = buildCollisionListForViz(cspaceGrid, gridSteps, qMin, qMax)
    print(f"Collision list has {len(cSpaceCollisions)} points.")

    # 5) Plan the path using A*
    pathSequence = []
    print("Planning path with A*...")
    for i in range(len(desiredIKs)):
        nxt = (i + 1) % len(desiredIKs)
        localPath = findPathAStar(cspaceGrid, desiredIKs[i], desiredIKs[nxt], qMin, qMax, gridSteps)
        if localPath is not None:
            # Avoid duplicate last config if not the final segment
            if i < len(desiredIKs) - 1:
                localPath.pop()
            pathSequence.extend(localPath)
        else:
            print(f"No path for segment {i}")
    print(f"A* complete. pathSequence length = {len(pathSequence)}")

    # 6) Interpolate path for smoother motion
    pathSequence = interpolatePath(pathSequence, stepsPerSegment=10)
    print(f"After interpolation, pathSequence length = {len(pathSequence)}")

    # ------------------- MATPLOTLIB FIGURE SETUP -------------------
    fig = plt.figure(figsize=(12, 6))

    # Left subplot: 3D C-space view
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title("C-space (collisions)")
    ax1.set_xlabel("θ1")
    ax1.set_ylabel("θ2")
    ax1.set_zlabel("θ3")
    q1_vals = [p[0] for p in cSpaceCollisions]
    q2_vals = [p[1] for p in cSpaceCollisions]
    q3_vals = [p[2] for p in cSpaceCollisions]
    ax1.scatter(q1_vals, q2_vals, q3_vals, c='red', s=2)
    ax1.set_xlim([qMin, qMax])
    ax1.set_ylim([qMin, qMax])
    ax1.set_zlim([qMin, qMax])

    # Right subplot: 2D workspace view
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("2D Simulation")
    ax2.set_aspect('equal', adjustable='box')

    # Draw ALL obstacles in the 2D workspace
    colors = ['r','orange','magenta']
    for idx, obs in enumerate(obstaculos):
        circ = plt.Circle((obs.center.x, obs.center.y),
                          obs.r, color=colors[idx % len(colors)], alpha=0.3)
        ax2.add_patch(circ)

    # Plot targets in blue
    for t in targets:
        ax2.plot(t.x, t.y, 'bo', markersize=6)

    # Draw the entire green path (effector positions)
    px, py = [], []
    savedQ1, savedQ2, savedQ3 = robot.q1, robot.q2, robot.q3
    for (q1, q2, q3) in pathSequence:
        robot.q1, robot.q2, robot.q3 = q1, q2, q3
        _, _, _, p3 = robot.getPoints()
        px.append(p3.x)
        py.append(p3.y)
    robot.q1, robot.q2, robot.q3 = savedQ1, savedQ2, savedQ3
    ax2.plot(px, py, 'g.-', label="Planned path")

    # Create line objects for the robot's links and joints (for animation)
    link1, = ax2.plot([], [], 'k-', linewidth=3)
    link2, = ax2.plot([], [], 'k-', linewidth=3)
    link3, = ax2.plot([], [], 'k-', linewidth=3)
    j0, = ax2.plot([], [], 'ko', markersize=6)
    j1, = ax2.plot([], [], 'ko', markersize=6)
    j2, = ax2.plot([], [], 'ko', markersize=6)
    j3, = ax2.plot([], [], 'ro', markersize=6)
    ax2.legend()

    # --------------------------------------------------------------------
    # Animation functions using FuncAnimation
    # --------------------------------------------------------------------
    def init():
        link1.set_data([], [])
        link2.set_data([], [])
        link3.set_data([], [])
        j0.set_data([], [])
        j1.set_data([], [])
        j2.set_data([], [])
        j3.set_data([], [])
        return link1, link2, link3, j0, j1, j2, j3

    skipFactor = 5
    def update(frame):
        realFrame = frame * skipFactor
        if realFrame >= len(pathSequence):
            return link1, link2, link3, j0, j1, j2, j3

        q1, q2, q3 = pathSequence[realFrame]
        robot.q1, robot.q2, robot.q3 = q1, q2, q3
        p0, p1, p2, p3 = robot.getPoints()

        link1.set_data([p0.x, p1.x], [p0.y, p1.y])
        link2.set_data([p1.x, p2.x], [p1.y, p2.y])
        link3.set_data([p2.x, p3.x], [p2.y, p3.y])
        j0.set_data([p0.x], [p0.y])
        j1.set_data([p1.x], [p1.y])
        j2.set_data([p2.x], [p2.y])
        j3.set_data([p3.x], [p3.y])

        return link1, link2, link3, j0, j1, j2, j3

    totalFrames = (len(pathSequence) + skipFactor - 1) // skipFactor
    ani = animation.FuncAnimation(
        fig, update,
        frames=totalFrames,
        init_func=init,
        interval=20,  # 20ms between frames (~50 FPS)
        blit=False
    )

    plt.show()

if __name__ == "__main__":
    main()
