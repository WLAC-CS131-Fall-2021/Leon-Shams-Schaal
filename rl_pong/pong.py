from pathlib import Path
from time import time

import numpy as np
import pygame

pygame.init()

# Set global variables
GAME_SPEED = 2

# Define colors
WHITE = [225, 225, 225]

PARENT_DIR = str(Path(__file__).parent) + "/"

# Load in sounds
master_channel = pygame.mixer.Channel(0)
master_channel.set_volume(0)

win_sound = pygame.mixer.Sound(PARENT_DIR + "Sound Effects/win-sound.wav")
lose_sound = pygame.mixer.Sound(PARENT_DIR + "Sound Effects/lose-sound.wav")
paddle_hit_sound = pygame.mixer.Sound(PARENT_DIR + "Sound Effects/paddle-hit.wav")
wall_bounce_sound = pygame.mixer.Sound(PARENT_DIR + "Sound Effects/wall-bounce.wav")


def scale_angle(angle):
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def ccw(p1, p2, p3):
    # Determine if the points are listed in a counterclockwise order
    return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])


def lines_intersect(lines):
    line1, line2 = lines
    return ccw(line1[0], *line2) != ccw(line1[1], *line2) and ccw(
        *line1, line2[0]
    ) != ccw(*line1, line2[1])


class Ball(pygame.Rect):
    direction = 0
    max_angle = np.pi / 3  # Steepest angle the ball can be hit at

    def __init__(self, size, speed, game):
        self.game = game
        self.set_speed(speed)

        size = self.game.min_dim * size
        self._x, self._y = (self.game.screen_width - size) / 2, (
            self.game.screen_height - size
        ) / 2
        super().__init__(self._x, self._y, size, size)
        self.previous_bounding_points = self.bounding_points()

        self.match_start_timestamp = time() + 1
        self.last_hit_paddle = None

    # `self.x` and `self.y` are rounded to the nearest whole number so
    # our own variables `self._x` and `self._y` are created to hold
    # the actual position of the ball down to the subpixel
    @property
    def _x(self):
        return self.__x

    @_x.setter
    def _x(self, __x):
        self.x = self.__x = __x

    @property
    def _y(self):
        return self.__y

    @_y.setter
    def _y(self, __y):
        self.y = self.__y = __y

    def bounding_points(self):
        return (
            (self._x, self._y),
            (self._x + self.width, self._y),
            (self._x, self._y + self.height),
            (self._x + self.width, self._y + self.height),
        )

    def set_speed(self, speed):
        # Default speed - Ball speed at the beginning of the match
        # Base speed - Speed of the ball after n hits (increases linearly with every hit)
        # Speed - The actual speed of the ball taking into account the number of hits and the angle of the current hit
        self.speed = self.base_speed = self.default_speed = (
            GAME_SPEED * self.game.screen_width * speed
        )

    def _check_for_collision(self, paddle):
        # To prevent the ball from phasing through the paddle we
        # track the path of the ball and check whether its path from
        # the previous step the current one instersects with the paddle
        current_bounding_points = self.bounding_points()
        ball_lines = list(zip(current_bounding_points, self.previous_bounding_points))

        paddle_corners = [
            paddle.topleft,
            paddle.topright,
            paddle.bottomleft,
            paddle.bottomright,
        ]
        paddle_lines = [
            (point1, point2)
            for point1, point2 in zip(
                paddle_corners, paddle_corners[-1:] + paddle_corners[:-1]
            )
        ]

        return any(
            [
                any(
                    [
                        lines_intersect([ball_line, paddle_line])
                        for paddle_line in paddle_lines
                    ]
                )
                for ball_line in ball_lines
            ]
        )

    def update(self, *paddles):
        self.previous_bounding_points = self.bounding_points()

        # Move the ball
        slow_down = 1 / 10 if time() - self.match_start_timestamp < 1 else 1
        self._x += slow_down * self.speed * np.cos(self.direction)
        self._y -= slow_down * self.speed * np.sin(self.direction)

        # Bounce the ball off the walls
        if self._y <= 0 or self._y >= self.game.screen_height - self.height:
            # The ball will overshoot the wall by a couple of pixels
            # To compensate the ball needs to be shifted back to the correct position
            # so it will bounce off the wall in the correct direction
            target_y = [self.game.screen_height - self.height, 0][int(self._y <= 0)]
            y_overshoot = target_y - self._y
            self._y += y_overshoot
            self._x -= y_overshoot / np.tan(self.direction)

            self.direction *= -1
            master_channel.play(wall_bounce_sound)

        # Sign of x-direction
        sign_direction = np.sign(np.cos(self.direction))
        # Sign of x-direction on the scale 0-1
        scaled_direction = int((sign_direction + 1) / 2)

        # Reset if there's a winner
        if self.x < -self.width or self.x > self.game.screen_width:
            self._x = (self.game.screen_width - self.width) / 2
            self._y = (self.game.screen_height - self.height) / 2
            self.previous_bounding_points = self.bounding_points()

            # Give the winning paddle a point
            winning_paddle = 1 - scaled_direction
            self.game.point_scored(paddles[winning_paddle])

            self.direction = np.pi * winning_paddle
            self.speed = self.base_speed = self.default_speed
            self.match_start_timestamp = time()

            return True

        # Bounce off paddle if we collide with a paddle
        for paddle in paddles:
            if paddle == self.last_hit_paddle:
                continue

            if self._check_for_collision(paddle):
                self.last_hit_paddle = paddle

                # Correct for the distance we went past the paddle
                x_overshoot = paddle.side - (self._x + scaled_direction * self.width)
                self._x += x_overshoot
                self._y -= x_overshoot * np.tan(self.direction)

                # Bounce ball off of paddle

                # Calculate how much we should redirect depending on where
                # on the paddle the ball hit
                hit_offset = (self.centery - paddle.centery) / paddle.height
                strike_redirect = sign_direction * (np.pi / 2) * hit_offset

                # Prevent the ball from going off in an angle steeper than `max_angle`
                clip_bounds = [-self.max_angle, self.max_angle]
                clip_bounds = scale_angle(np.multiply(sign_direction, clip_bounds))

                # Bounce
                self.direction = np.pi - self.direction + strike_redirect

                # Clip direction
                shift = np.pi * (clip_bounds[0] < clip_bounds[1])
                self.direction = (
                    np.clip(scale_angle(self.direction - shift), *sorted(clip_bounds))
                    + shift
                )

                # Increase ball speed
                self.base_speed += GAME_SPEED * self.game.screen_width * 0.000075
                self.speed = self.base_speed * (1 + abs(hit_offset / 2))

                master_channel.play(paddle_hit_sound)

                # Take next step because it makes the bounces look more natural
                self.update(*paddles)

        return False

    def project_pos(self, paddle):
        # Predicts the height of the ball when it reaches the paddle

        playable_height = self.game.screen_height - self.height
        ball_x = self._x + (np.sign(np.cos(self.direction)) / 2 + 0.5) * self.width
        ball_y = self._y + self.height / 2

        if not self.direction:
            return ball_y

        shifted_path = (
            np.tan(-self.direction)
            * (paddle.side - ball_x + ball_y / np.tan(-self.direction))
            - self.height / 2
        )
        scaled_path = np.pi * shifted_path / playable_height
        target_y = (playable_height / np.pi) * np.arccos(
            np.cos(scaled_path)
        ) + self.height / 2
        return target_y


class Paddle(pygame.Rect):
    points = 0

    def __init__(
        self, x_pos, acceleration, min_speed, max_speed, ball, get_color, game
    ):
        self.acceleration = acceleration
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.reset_velocity()

        self.ball = ball
        self.get_color = get_color

        self.game = game
        dims = (0.02 * self.game.min_dim, 0.1 * self.game.min_dim)
        pos = [
            x_pos * self.game.screen_width - dims[0] / 2,
            0,  # Just a placeholder. We'll vertically center soon
        ]
        super().__init__(pos, dims)
        self.vertically_center()

    # Because `self.y` is rounded to the nearest integer we use `self._y` to
    # store the y position of the paddle down to the subpixel
    @property
    def _y(self):
        return self.__y

    @_y.setter
    def _y(self, __y):
        self.y = self.__y = __y

    @property
    def side(self):
        # Return side of the paddle that will hit the ball
        return self.right if self.centerx < self.game.screen_width / 2 else self.left

    @property
    def speed(self):
        return (
            self.velocity
            * self.ball.base_speed
            * self.game.screen_height
            / self.game.screen_width
        )

    def set_speed(self, min_speed, max_speed):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.reset_velocity()

    def reset_velocity(self):
        self.velocity = self.min_speed

    def vertically_center(self):
        self._y = (self.game.screen_height - self.height) / 2

    def move(self, direction):
        self.velocity += self.acceleration
        self.velocity = min(self.velocity, self.max_speed)

        self._y -= direction * self.speed

        # Prevent paddle from going off the screen
        self._y = np.clip(
            self._y,
            -self.height / 2,
            self.game.screen_height - self.height / 2,
        )


def load_img(img_path, size, pos):
    img = pygame.image.load(PARENT_DIR + img_path)

    # Make image slightly transparent
    img.convert_alpha()
    img.fill((*WHITE, WHITE[-1]), None, pygame.BLEND_RGBA_MULT)

    # Position image
    img = pygame.transform.rotozoom(img, 0, size / img.get_size()[0])
    rect = img.get_rect(center=pos)
    return img, rect


class Pong:
    background_color = [0, 100, 0]

    start_ball_speed, end_ball_speed = 0.0015, 0.0065
    start_opponent_speed, end_opponent_speed = 0.15, 0.61
    start_controlled_paddle_speed, end_controlled_paddle_speed = 2.5, 0.61

    difficulty_level = 0
    controlled_paddle_direction = 0  # Remembers last user key press
    opponent_paddle_hit_edge = 0
    running = True

    def __init__(self, screen_dims=None, reset_paddle_pos=True, fps=None):
        self.reset_paddle_pos = reset_paddle_pos
        self.fps = fps

        # Create screen
        fullscreen = screen_dims is None
        pygame.mouse.set_visible(not fullscreen)

        if fullscreen:
            # Create a fullscreen display
            display_info = pygame.display.Info()
            screen_dims = display_info.current_w, display_info.current_h
            display_options = (
                pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SCALED
            )
        else:
            # Create screen of specified dimensions
            display_options = pygame.SCALED

        self.screen = pygame.display.set_mode(screen_dims, display_options, vsync=1)
        self.clock = pygame.time.Clock()

        self.screen_width, self.screen_height = (
            self.screen.get_width(),
            self.screen.get_height(),
        )
        self.min_dim = min(self.screen_width, self.screen_height)

        # Create play button
        play_button_size = 0.15 * self.min_dim
        play_button_pos = self.screen_width / 2, self.screen_height / 2
        self.play_img, self.play_button = load_img(
            "Images/play.png", play_button_size, play_button_pos
        )
        self.play_button_displayed = False

        # Create pause button
        pause_button_size = 0.06 * self.min_dim
        pause_button_pos = self.screen_width / 2, self.screen_height - pause_button_size
        self.pause_img, self.pause_button = load_img(
            "Images/pause.png", pause_button_size, pause_button_pos
        )
        self.paused = False

        # Load in fonts
        score_font_size = self.min_dim // 10
        self.score_font = pygame.font.Font(
            PARENT_DIR + "Fonts/roboto-mono.ttf", score_font_size
        )

        self.difficulty_font_size = self.min_dim // 30
        self.difficulty_font = pygame.font.Font(
            PARENT_DIR + "Fonts/roboto-mono.ttf", self.difficulty_font_size
        )

        # Create game objects
        self.ball = Ball(size=0.02, speed=self.start_ball_speed, game=self)
        self.opponent_paddle = Paddle(
            x_pos=0.1,
            acceleration=0,
            min_speed=self.start_opponent_speed,
            max_speed=self.start_opponent_speed,
            ball=self.ball,
            get_color=self._opponent_color,
            game=self,
        )
        self.controlled_paddle = Paddle(
            x_pos=0.9,
            acceleration=0,  # 0.01,
            min_speed=self.start_controlled_paddle_speed,  # 1,
            max_speed=self.start_controlled_paddle_speed,  # 1.3,
            ball=self.ball,
            get_color=lambda: WHITE,
            game=self,
        )

    def _opponent_color(self):
        return np.multiply(self.background_color, 1.3)

    def calculate_difficulty(self):
        return -np.e ** (-self.difficulty_level / 10) + 1

    def _render_score(self, score, x_pos, color):
        text_surface = self.score_font.render(str(score), True, color)
        text_rect = text_surface.get_rect(center=[x_pos, self.screen_height / 10])
        self.screen.blit(text_surface, text_rect)

    def point_scored(self, paddle):
        # Update the score and difficulty whenever a point is scored
        paddle.points += 1

        if self.reset_paddle_pos:
            self.opponent_paddle.vertically_center()
            self.controlled_paddle.vertically_center()

        if paddle == self.controlled_paddle:
            master_channel.play(win_sound)
            self.difficulty_level += 1
        else:
            master_channel.play(lose_sound)
            self.difficulty_level = max(0, self.difficulty_level - 1)

        difficulty = self.calculate_difficulty()

        # Change background color based on difficulty
        self.background_color[0] = 100 * difficulty
        self.background_color[1] = 100 * (1 - difficulty)

        # Update game to new difficulty
        opponent_speed = np.interp(
            difficulty, (0, 1), (self.start_opponent_speed, self.end_opponent_speed)
        )
        self.opponent_paddle.set_speed(opponent_speed, opponent_speed)

        controlled_paddle_speed = np.interp(
            difficulty,
            (0, 1),
            (self.start_controlled_paddle_speed, self.end_controlled_paddle_speed),
        )
        self.controlled_paddle.set_speed(
            controlled_paddle_speed, controlled_paddle_speed
        )

        self.ball.set_speed(
            np.interp(difficulty, (0, 1), (self.start_ball_speed, self.end_ball_speed))
        )

    @property
    def num_games(self):
        return self.opponent_paddle.points + self.controlled_paddle.points

    def user_interaction(self, enable_mouse_control=False):
        for event in pygame.event.get():
            # Quit if we hit the x-button on the window or use a shortcut
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Handle pausing
                mouse_pos = pygame.mouse.get_pos()
                if self.paused:
                    if self.play_button.collidepoint(mouse_pos):
                        self.paused = False
                        self.play_button_displayed = False
                else:
                    if self.pause_button.collidepoint(mouse_pos):
                        self.paused = True

            # Mouse based movement
            if event.type == pygame.MOUSEMOTION and enable_mouse_control:
                self.controlled_paddle_direction = 0
                self.controlled_paddle._y = pygame.mouse.get_pos()[1]

            elif event.type == pygame.KEYDOWN:
                # Quit if the escape key is pressed
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                # Pause if "p" is pressed
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                    if not self.paused:
                        self.play_button_displayed = False

                # Move up
                if event.key in [pygame.K_k, pygame.K_UP]:
                    self.controlled_paddle_direction = 1

                # Move down
                elif event.key in [pygame.K_m, pygame.K_DOWN]:
                    self.controlled_paddle_direction = -1

            elif event.type == pygame.KEYUP:
                # Stop moving up
                if event.key in [pygame.K_k, pygame.K_UP]:
                    if self.controlled_paddle_direction == 1:
                        self.controlled_paddle_direction = 0

                # Stop moving down
                elif event.key in [pygame.K_m, pygame.K_DOWN]:
                    if self.controlled_paddle_direction == -1:
                        self.controlled_paddle_direction = 0

        return self.controlled_paddle_direction + 1

    @property
    def state(self):
        opponent_y = self.opponent_paddle.centery / self.screen_height
        paddle_y = self.controlled_paddle.centery / self.screen_height
        ball_pos = np.divide(self.ball.center, [self.screen_width, self.screen_height])
        ball_direction = (self.ball.direction + np.pi) / (2 * np.pi)
        return np.array(
            [opponent_y, paddle_y, *ball_pos, ball_direction], dtype=np.float32
        )

    @property
    def reward(self):
        """Reward scheme based on distance from the ball hit target"""

        # Ball is moving away from us
        if np.sign(np.cos(self.ball.direction)) == -1:
            return 1.25

        # Ball is moving toward us
        target_y = self.ball.project_pos(self.controlled_paddle)
        dist_from_target = abs(target_y - self.controlled_paddle.centery)
        dist_from_target = max(dist_from_target - self.controlled_paddle.height / 2, 0)

        scaled_target_dist = (
            self.screen_height - dist_from_target
        ) / self.screen_height

        # We square the dist to encourage adjustments when near the target
        return scaled_target_dist ** 2

    def step(self, action):
        # No need to update if the screen is paused
        if self.play_button_displayed or not self.running:
            if self.fps:
                self.clock.tick(self.fps)
            return None, None, None

        # Calculate the ideal position for the AI
        if np.sign(np.cos(self.ball.direction)) == -1:
            if not self.opponent_paddle_hit_edge:
                self.opponent_paddle_hit_edge = 1 if np.random.random() < 0.5 else -1

            projected_y = self.ball.project_pos(self.opponent_paddle)
        else:
            self.opponent_paddle_hit_edge = 0
            projected_y = self.screen_height / 2

        projected_y += self.opponent_paddle_hit_edge * self.opponent_paddle.height / 2

        # Set `target_y` to the weighted average between the
        # ideal position (`projected_y`) and the height of the ball
        difficulty = self.calculate_difficulty()
        target_y = difficulty * projected_y + (1 - difficulty) * self.ball.centery

        # Move the opponent paddle
        if target_y > self.opponent_paddle.centery + np.ceil(
            self.opponent_paddle.speed
        ):
            self.opponent_paddle.move(-1)
        elif target_y < self.opponent_paddle.centery:
            self.opponent_paddle.move(1)

        # Move player paddle
        action -= 1
        if action:
            self.controlled_paddle.move(action)
        else:
            self.controlled_paddle.reset_velocity()

        self.screen.fill((self.background_color))

        # Render pause and play buttons
        if not self.paused:
            self.screen.blit(self.pause_img, self.pause_button)
        else:
            self.screen.blit(self.play_img, self.play_button)
            self.play_button_displayed = True

        # Render text
        self._render_score(
            self.opponent_paddle.points,
            0.25 * self.screen_width,
            self._opponent_color(),
        )
        self._render_score(
            self.controlled_paddle.points,
            0.75 * self.screen_width,
            WHITE,
        )

        difficulty_percentage = self.difficulty_font.render(
            f"{100 * difficulty:2.0f}%", True, self._opponent_color()
        )
        self.screen.blit(
            difficulty_percentage,
            dest=[
                self.difficulty_font_size,
                self.screen_height - 2 * self.difficulty_font_size,
            ],
        )

        # Render paddles
        for paddle in (self.opponent_paddle, self.controlled_paddle):
            pygame.draw.rect(self.screen, paddle.get_color(), paddle)

        # Render and update ball
        pygame.draw.rect(self.screen, WHITE, self.ball)
        done = self.ball.update(self.opponent_paddle, self.controlled_paddle)

        # Refresh display
        pygame.display.flip()
        if self.fps:
            self.clock.tick(self.fps)

        return self.state, self.reward, done

    def finish(self):
        self.running = False
        pygame.quit()
