import pygame
from pygame.locals import *
import neat
import time
import os
import random
pygame.font.init()

WINDOW_WIDTH = 550
WINDOW_HEIGHT = 800

GENERATION = 0

IMAGES_BIRD = [
    pygame.transform.scale2x(pygame.image.load(os.path.join("sprites", "bird1.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("sprites", "bird2.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("sprites", "bird3.png")))
]
IMAGE_PIPE = pygame.transform.scale2x(pygame.image.load(os.path.join("sprites", "pipe.png")))
IMAGE_BG = pygame.transform.scale2x(pygame.image.load(os.path.join("sprites", "bg.png")))
IMAGE_GROUND = pygame.transform.scale2x(pygame.image.load(os.path.join("sprites", "base.png")))

STAT_FONT = pygame.font.SysFont("comicsans", 50)


class Bird:
    IMAGES = IMAGES_BIRD
    MAX_ROTATION = 25 # How much to Tilt
    ROTATION_VELOCITY = 20 # How much to rotate each frame
    ANIMATION_TIME = 5 # How long

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tickCount = 0
        self.velocity = 0
        self.height = self.y
        self.imageCount = 0
        self.image = self.IMAGES[0]
    
    def jump(self):
        self.velocity = -10.5
        self.tickCount = 0
        self.height = self.y

    def move(self):
        self.tickCount += 1

        # Gravity
        displacement = self.velocity*self.tickCount + 1.5*self.tickCount**2

        if displacement>= 16:
            displacement = 16
        if displacement<= 0:
            displacement -= 2

        self.y = self.y + displacement
        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROTATION_VELOCITY

    def draw(self, window):
        self.imageCount += 1


        if self.imageCount < self.ANIMATION_TIME:
            self.image = self.IMAGES[0]
        elif self.imageCount < self.ANIMATION_TIME*2:
            self.image = self.IMAGES[1]
        elif self.imageCount < self.ANIMATION_TIME*3:
            self.image = self.IMAGES[2]
        elif self.imageCount < self.ANIMATION_TIME*4:
            self.image = self.IMAGES[1]
        elif self.imageCount == self.ANIMATION_TIME*4 + 1:
            self.image = self.IMAGES[0]
            self.imageCount = 0

        if self.tilt <= -80:
            self.img = self.IMAGES[1]
            self.imageCount = self.ANIMATION_TIME*2

        rotatedImage = pygame.transform.rotate(self.image, self.tilt)
        newRect = rotatedImage.get_rect(center=self.image.get_rect(topleft = (self.x, self.y)).center)
        window.blit(rotatedImage, newRect.topleft)
    
    def getMask(self):
        return pygame.mask.from_surface(self.image)

class Pipe:
    GAP = 200
    VELOCITY = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(IMAGE_PIPE, False, True)
        self.PIPE_BOTTOM = IMAGE_PIPE

        self.passed = False
        self.setHeight()

    def setHeight(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height();
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VELOCITY

    def draw(self, window):
        window.blit(self.PIPE_TOP, (self.x, self.top))
        window.blit(self.PIPE_BOTTOM, (self.x, self.bottom))
        
    def collide(self, bird):
        birdMask = bird.getMask()
        topMask = pygame.mask.from_surface(self.PIPE_TOP)
        bottomMask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        topOffset = (self.x - bird.x, self.top - round(bird.y))
        bottomOffset = (self.x - bird.x, self.bottom - round(bird.y))

        bottomPoint = birdMask.overlap(bottomMask, bottomOffset)
        topPoint = birdMask.overlap(topMask, topOffset)
        
        if topPoint or bottomPoint:
            return True
        return False

class Ground:
    VELOCITY = 5
    WIDTH = IMAGE_GROUND.get_width()
    IMAGE = IMAGE_GROUND

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VELOCITY
        self.x2 -= self.VELOCITY

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, window):
        window.blit(self.IMAGE, (self.x1, self.y)) 
        window.blit(self.IMAGE, (self.x2, self.y)) 

def drawWindow(window, birds, pipes, ground, score, generation):
    window.blit(IMAGE_BG, (0,0))

    for pipe in pipes:
        pipe.draw(window)

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    window.blit(text, (WINDOW_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Gen: " + str(generation), 1, (255, 255, 255))
    window.blit(text, (10, 10))

    text = STAT_FONT.render("# of Birds: " + str(len(birds)), 1, (255, 255, 255))
    window.blit(text, (10, 50))

    ground.draw(window)
    for bird in birds:
        bird.draw(window)
    pygame.display.update()

def mainLoop(genomes, config):

    global GENERATION
    GENERATION += 1

    nets = []
    genomeList = []
    birds = []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        genome.fitness = 0
        genomeList.append(genome)

    ground = Ground(730)
    pipes = [Pipe(600)]
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    clock = pygame.time.Clock()
    score = 0
    run = True
    while run:
        clock.tick(40)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipeIdx = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipeIdx = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            genomeList[x].fitness += 0.1

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipeIdx].height), abs(bird.y - pipes[pipeIdx].bottom)))
            
            if output[0] > 0.5:
                bird.jump()

        removeArr = []
        addPipe = False
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    genomeList[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    genomeList.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    addPipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                removeArr.append(pipe)
            
            pipe.move()

        if addPipe:
            score += 1
            for genome in genomeList:
                genome.fitness += 1
            pipes.append(Pipe(600))

        for r in removeArr:
            pipes.remove(r)
        for x, bird in enumerate(birds):
            if bird.y + bird.image.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                genomeList.pop(x)

        ground.move()
        drawWindow(window, birds, pipes, ground, score, GENERATION)


def run(configPath):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, configPath)

    p = neat.Population(config)

    # Output
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(mainLoop,50)

if __name__ == "__main__":
    localDir = os.path.dirname(__file__)
    configPath = os.path.join(localDir, "config-feedforward.txt")
    run(configPath)