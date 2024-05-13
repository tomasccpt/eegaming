#include "c.h"
#include "raylib.h"

enum CONSTANTS {
	WINDOW_WIDTH = 1280,
	WINDOW_HEIGHT = 720,
	TARGET_FRAMERATE = 120,

	LEVEL_WIDTH = 16,
	LEVEL_HEIGHT = 9,

	SQUARE_SIDE = WINDOW_WIDTH / LEVEL_WIDTH,
};

#define PHYSICS_DELTA 0.150f

static i32 current_level;
static i32 player_x_old;
static i32 player_y_old;
static i32 player_x;
static i32 player_y;
static i32 player_dx;
static i32 player_dy;
static f32 accumulator = 0.0f;

static u8 (*current_map)[LEVEL_HEIGHT][LEVEL_WIDTH];

static u8 maps[][LEVEL_HEIGHT][LEVEL_WIDTH] = { 
	{
		[0][4] = 2,
		[0][3] = 1,
	},
	{
		[3][0] = 2,
		[0][3] = 1,
		[2][0] = 1,
		[1][11] = 1,
		[7][0] = 1,
		[7][15] = 1,
		[8][4] = 1,
	},
};
static u8 initial_pos[][2] = {
	{0, 0},
	{8, 1},
};

/* Returns if exited due to window close. */
static u32 renderGameCompleted(void);

int main(void) {
	static_assert(len(maps) == len(initial_pos), "Len of maps does not match the len of initial positions.");

	InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "EEGaming");
	SetTargetFPS(TARGET_FRAMERATE);

	current_map = &(maps[current_level]);
	player_y = initial_pos[current_level][0];
	player_x = initial_pos[current_level][1];

	while (!WindowShouldClose()) {
		if (player_dx == 0 && player_dy == 0) {
			if (IsKeyPressed(KEY_W)) {
				player_dy = -1;
				player_dx = 0;
			} else if (IsKeyPressed(KEY_S)) {
				player_dy = 1;
				player_dx = 0;
			} else if (IsKeyPressed(KEY_A)) {
				player_dx = -1;
				player_dy = 0;
			} else if (IsKeyPressed(KEY_D)) {
				player_dx = 1;
				player_dy = 0;
			}
			accumulator = PHYSICS_DELTA;
		} else {
			float ft = GetFrameTime();
			if (ft > PHYSICS_DELTA) {
				ft = PHYSICS_DELTA;
			}
			accumulator += ft;
		}

		while (accumulator >= PHYSICS_DELTA) {
			player_y_old = player_y;
			player_x_old = player_x;
			player_y += player_dy;
			player_x += player_dx;
			accumulator -= PHYSICS_DELTA;
			if (player_x < 0) {
				player_x = 0;
				player_dx = 0;
			} else if (player_x >= LEVEL_WIDTH) {
				player_x = LEVEL_WIDTH - 1;
				player_dx = 0;
			} else if (player_y < 0) {
				player_y = 0;
				player_dy = 0;
			} else if (player_y >= LEVEL_HEIGHT) {
				player_y = LEVEL_HEIGHT - 1;
				player_dy = 0;
			} else if ((*current_map)[player_y][player_x] == 1) {
				player_y = player_y_old;
				player_x = player_x_old;
				player_dy = 0;
				player_dx = 0;
			} else if ((*current_map)[player_y][player_x] == 2) {
				current_level += 1;
				if (current_level == len(maps)){
					/* WindowShoudClose() just works once, so passing up if user wants to close. */
					if (renderGameCompleted()) {
						return 0;
					};
					current_level = 0;
				}
				current_map = &(maps[current_level]);
				player_y = initial_pos[current_level][0];
				player_x = initial_pos[current_level][1];
				player_dx = 0;
				player_dy = 0;
			}
		}


		BeginDrawing();
		ClearBackground((Color){.r = 0x18, .g = 0x18, .b = 0x18, .a = 0xff});
		for (u32 y = 0; y < LEVEL_HEIGHT; ++y) {
			for (u32 x = 0; x < LEVEL_WIDTH; ++x) {
				if ((*current_map)[y][x] == 1) {
					DrawRectangle(x * SQUARE_SIDE, y * SQUARE_SIDE, SQUARE_SIDE, SQUARE_SIDE, RAYWHITE);
				} else if ((*current_map)[y][x] == 2) {
					DrawRectangle(x * SQUARE_SIDE, y * SQUARE_SIDE, SQUARE_SIDE, SQUARE_SIDE, RED);
				}
			}
		}
		if (player_dx == 0 && player_dy == 0) {
			DrawRectangle(player_x * SQUARE_SIDE, player_y * SQUARE_SIDE, SQUARE_SIDE, SQUARE_SIDE, BLUE);
		} else {
			int x = (((accumulator / PHYSICS_DELTA) * (player_x - player_x_old)) + player_x_old) * SQUARE_SIDE;
			int y = (((accumulator / PHYSICS_DELTA) * (player_y - player_y_old)) + player_y_old) * SQUARE_SIDE;
			DrawRectangle(x, y, SQUARE_SIDE, SQUARE_SIDE, BLUE);
		}
		EndDrawing();
	}

	return 0;
}


static u32 renderGameCompleted(void) {
	enum {
		TITLE_SIZE = 96,
		BOTTOM_TEXT_SIZE = 32,
	};
	char *title = "Game completed!";
	char *bottom_text = "Press R to replay.";

	int title_width = MeasureText(title, TITLE_SIZE);
	int bottom_text_width = MeasureText(bottom_text, BOTTOM_TEXT_SIZE);

	int title_x = (WINDOW_WIDTH - title_width) / 2;
	int bottom_text_x = (WINDOW_WIDTH - bottom_text_width) / 2;

	while (!WindowShouldClose()) {
		if (IsKeyPressed(KEY_R)) {
			return 0;
		}

		BeginDrawing();
		ClearBackground((Color){.r = 0x18, .g = 0x18, .b = 0x18, .a = 0xff});
		DrawText("Game completed!", title_x, 300, TITLE_SIZE, RAYWHITE);
		DrawText("Press R to replay.", bottom_text_x, 450, BOTTOM_TEXT_SIZE, RAYWHITE);
		EndDrawing();
	}

	return 1;
}
