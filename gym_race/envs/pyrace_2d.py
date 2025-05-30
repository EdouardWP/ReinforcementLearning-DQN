import pygame
import math
import numpy as np

screen_width = 1500
screen_height = 800
check_point = ((1370, 675), (1370, 215), (935, 465), (630, 180), (320, 160), (130, 675), (550, 702)) # race_track_ie.png
"""
Area for text boxes:
750,0 - 1500,100
1055,290 - 1300,605
"""

class Car:
    def __init__(self, car_file, map, pos): # map_file
        # self.map = pygame.image.load(map_file)
        self.map = map
        self.surface = pygame.image.load(car_file)
        self.surface = pygame.transform.scale(self.surface, (100, 100))
        self.rotate_surface = self.surface
        self.pos = pos
        self.angle = 0
        self.speed = 0
        self.center = [self.pos[0] + 50, self.pos[1] + 50]
        self.radars = []
        self.radars_for_draw = []
        self.is_alive = True
        self.goal = False
        self.distance = 0
        self.time_spent = 0
        self.current_check = 0
        self.prev_distance = 0
        self.cur_distance = 0
        self.check_flag = False
        
        # Initialize radars
        for d in range(-90, 120, 45):
            self.check_radar(d)

    def draw(self, screen):
        screen.blit(self.rotate_surface, self.pos)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for r in self.radars: # or self.radars_for_draw
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def pixel_at(self,x,y):
        try:
            return self.map.get_at((x,y))
        except:
            return (255, 255, 255, 255)

    def check_collision(self, map=None):
        self.is_alive = True
        for p in self.four_points:
            if self.pixel_at(int(p[0]), int(p[1])) == (255, 255, 255, 255):
                self.is_alive = False
                break

    def check_radar(self, degree, map=None):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not self.pixel_at(x, y) == (255, 255, 255, 255) and len < 200:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])
    """
    #------------------------------------------------------------------------------
    def draw_collision(self, screen):
        for i in range(4):
            x = int(self.four_points[i][0])
            y = int(self.four_points[i][1])
            pygame.draw.circle(screen, (255, 255, 255), (x, y), 5)

    def check_radar_for_draw(self, degree, map=None):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not self.map.get_at((x, y)) == (255, 255, 255, 255) and len < 2000:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars_for_draw.append([(x, y), dist])
    """
    def check_checkpoint(self):
        p = check_point[self.current_check]
        self.prev_distance = self.cur_distance
        dist = get_distance(p, self.center)
        if dist < 70:
            self.current_check += 1
            self.prev_distance = 9999
            self.check_flag = True
            if self.current_check >= len(check_point):
                self.current_check = 0
                self.goal = True
            else:
                self.goal = False

        self.cur_distance = dist
    #------------------------------------------------------------------------------


    def update(self,map=None):
        #check speed
        self.speed -= 0.3  # Reduced friction for smoother deceleration
        if self.speed > 10: self.speed = 10
        if self.speed < 0:  self.speed = 0  # Allow car to come to a complete stop
        
        # required for NEAT
        if map is not None:
            self.speed = 7 # NEAT

        #check position
        self.rotate_surface = self.rot_center(self.surface, self.angle)
        self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        if self.pos[0] < 20:
            self.pos[0] = 20
        elif self.pos[0] > screen_width - 120:
            self.pos[0] = screen_width - 120

        self.distance += self.speed
        self.time_spent += 1
        self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        if self.pos[1] < 20:
            self.pos[1] = 20
        elif self.pos[1] > screen_height - 120:
            self.pos[1] = screen_height - 120

        # caculate 4 collision points
        self.center = [int(self.pos[0]) + 50, int(self.pos[1]) + 50]
        len = 40
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * len,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * len]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * len,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * len]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * len,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * len]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * len,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * len]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]

        # required for NEAT
        if map is not None:
            self.check_collision(self.map)
            self.radars.clear()
            for d in range(-90, 120, 45):
                self.check_radar(d, self.map)

        """
        self.car.radars_for_draw.clear()
        for d in range(-90, 105, 15):
            self.car.check_radar_for_draw(d)
        pygame.draw.circle(self.screen, (255, 255, 0), check_point[self.car.current_check], 70, 1)
        
        self.car.draw_collision(self.screen)
        # self.car.draw_radar(self.screen) # moved to car.draw()
        self.car.draw(self.screen)
        """

    #-------------------------------------------------------------------
    # required for NEAT
    def get_data(self):
        radars = self.radars
        ret = [0, 0, 0, 0, 0]
        for i, r in enumerate(radars):
            ret[i] = int(r[1] / 30)
        return ret

    def get_alive(self):
        return self.is_alive

    def get_reward(self):
        return self.distance / 50.0
    #-------------------------------------------------------------------

    def rot_center(self, image, angle):
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image


class PyRace2D:
    def __init__(self, is_render = True, car = True, mode = 0, continuous_radar=False):
        # print('PyRace2D - INIT ENVIRONMENT')
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)
        self.map = pygame.image.load('race_track_ie.png')
        self.cars = []
        if car:
            self.car = Car('car.png', self.map, [500, 650])
            self.cars.append(self.car)
        self.game_speed = 60*0 # as fast as possible...
        self.is_render = is_render
        self.mode = mode # 0: normal, 1:dark, 2: normal (force display)
        self.continuous_radar = continuous_radar  # Flag for continuous radar values

    def action(self, action):
        # For discrete action space (compatibility with v1)
        if isinstance(action, int):
            if action == 0: self.car.speed += 2
            elif action == 1: self.car.angle += 5
            elif action == 2: self.car.angle -= 5
            elif action == 3: self.car.speed -= 2  # New BRAKE action
        # For continuous action space
        elif isinstance(action, (list, tuple, np.ndarray)) and len(action) == 2:
            # action[0]: acceleration (-1 to 1), action[1]: steering (-1 to 1)
            self.car.speed += action[0] * 2  # Scale to similar range
            self.car.angle += action[1] * 5  # Scale to similar range

        self.car.update()
        self.car.check_collision()
        self.car.check_checkpoint()

        self.car.radars.clear()
        for d in range(-90, 120, 45):
            self.car.check_radar(d)

    def evaluate(self):
        reward = 0
        
        # Check for checkpoint progress
        if self.car.check_flag:
            self.car.check_flag = False
            # Higher reward for reaching checkpoints quickly
            reward += 200  # Increased checkpoint reward
            
        # Progressive reward based on distance to next checkpoint
        current_checkpoint = check_point[self.car.current_check]
        distance_to_checkpoint = get_distance(current_checkpoint, self.car.center)
        
        # Reward for getting closer to the next checkpoint - more aggressive rewards
        if self.car.prev_distance > distance_to_checkpoint:
            reward += 0.5 * (self.car.prev_distance - distance_to_checkpoint)  # 5x stronger reward
        
        # Penalty for moving away from checkpoint
        elif self.car.prev_distance < distance_to_checkpoint:
            reward -= 0.1 * (distance_to_checkpoint - self.car.prev_distance)
        
        # Stronger living bonus to encourage survival
        reward += 0.1

        # Speed-based reward component (encourage moderate speed)
        optimal_speed = 7
        speed_reward = 1.0 - abs(self.car.speed - optimal_speed) / 10.0
        reward += speed_reward * 0.5  # Increased speed reward importance
            
        # Failure cases
        if not self.car.is_alive:  # crash
            reward = -20  # Less severe crash penalty
        elif self.car.goal:  # reached final checkpoint
            reward = 1000  # Higher goal reward
            
        return reward

    def is_done(self):
        if not self.car.is_alive or self.car.goal:
            self.car.current_check = 0
            self.car.distance = 0
            return True
        return False

    def observe(self):
        # return state
        radars = self.car.radars
        radar_values = []
        
        # Make sure we have 5 radar values
        if len(radars) < 5:
            # Initialize radars if they're empty
            self.car.radars.clear()
            for d in range(-90, 120, 45):
                self.car.check_radar(d)
            radars = self.car.radars
        
        for i, r in enumerate(radars):
            if self.continuous_radar:
                # Return actual distance values (normalized by 200, which is max radar length)
                radar_values.append(r[1] / 200.0)
            else:
                # Original bucketed values (for backward compatibility)
                radar_values.append(int(r[1] / 30))
        
        return radar_values

    def view_(self, msgs=[]): # RENDERING...
        # draw game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.mode += 1
                    self.mode = self.mode % 3
                if event.key == pygame.K_p:
                    self.mode += 1
                    self.mode = self.mode % 3
                elif event.key == pygame.K_q:
                    done = True
                    exit()

        self.screen.blit(self.map, (0, 0))

        if self.mode == 1:
            self.screen.fill((0, 0, 0))
        """
        self.car.radars_for_draw.clear()
        for d in range(-90, 105, 15):
            self.car.check_radar_for_draw(d)
        """
        if len(self.cars) == 1:
            pygame.draw.circle(self.screen, (255, 255, 0), check_point[self.car.current_check], 70, 1)
        """
        self.car.draw_collision(self.screen)
        """
        # self.car.draw_radar(self.screen) # moved to car.draw()
        
        # self.car.draw(self.screen)
        for car in self.cars:
            if car.get_alive():
                car.draw(self.screen)

        # Display messages...
        for k,msg in enumerate(msgs):
            myfont = pygame.font.SysFont("impact", 20)
            label = myfont.render(msg, 1, (0,0,0))
            self.screen.blit(label,(1055,290+k*25))
            pass

        text = self.font.render("Press 'm' to change view mode", True, (255, 255, 0))
        text_rect = text.get_rect()
        # text_rect.center = (screen_width/2, 100)
        text_rect.topleft = (750,0)
        self.screen.blit(text, text_rect)

        pygame.display.flip()
        self.clock.tick(self.game_speed)


def get_distance(p1, p2):
	return math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))
