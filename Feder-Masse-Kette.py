"""
Feder–Masse–Kette mit pymunk/pygame

Anforderungen:
- N=100 Massen (m=0.1 kg), horizontal angeordnet.
- Jede Masse ist mit Vorgänger/Nachfolger über Federn (Ruhelänge l=0.1 m) verbunden.
- Die letzte Feder ist an einer vertikalen Wand (rechter Rand) befestigt.
- Die erste Feder wird kurz nach dem Start schlagartig um 0.5 m nach oben bewegt.
- Gravitation g und Reibung (Dämpfung) sind einstellbar; initial g=0.

Hinweis zu Einheiten: Wir verwenden einen Skalierungsfaktor Meter→Pixel, so dass die
physikalischen Längenangaben in Metern als Pixel auf dem Bildschirm dargestellt werden.
"""

from __future__ import annotations

import math
import pygame
import pymunk
import pymunk.pygame_util


# -----------------------------
# Konfigurierbare Parameter
# -----------------------------

# Grundlegendes
N_MASSES = 150              # Anzahl Massen
MASS_KG = 0.05               # Masse je Körper [kg]

# Federlängen
L0_M = 0.1                  # entspannte Federlänge (Ruhelänge) l0 [m]
LV_M = 0.12                 # Vorspannungslänge lv (Start-Abstand pro Segment) [m]

PIXELS_PER_METER = 50.0     # Skalierung [px/m]

# Federparameter
SPRING_STIFFNESS = 1200.0   # Federsteifigkeit (Pymunk-Einheiten)
SPRING_DAMPING = 8.0       # Federdämpfung 8

# Gravitation und Reibung (Dämpfung)
GRAVITY = (0.0, 0.0)      # initial g=0; z.B. (0, 9.81*PIXELS_PER_METER)
SPACE_DAMPING = 1.0       # =1.0: keine Dämpfung, <1.0: Dämpfung
                          # space.damping ist ein Multiplikationsfaktor 
                          # für die Geschwindigkeit pro Integrationsschritt

# Impuls der ersten Feder (linkes Ende)
IMPULSE_TIME_S = 0.25       # Zeitpunkt nach Start [s]
ANCHOR_LIFT_M = 0.5         # Sprung nach oben [m]

# Fenster / Darstellung
WIDTH, HEIGHT = 1200, 600
FPS_LIMIT = 60
MASS_RADIUS_PX = 2          # Darstellungsradius der Massen [px]
LINE_COLOR = (30, 144, 255) # Farbe der Federlinien (DodgerBlue)
MASS_COLOR = (10, 10, 10)
BG_COLOR = (245, 245, 245)
WALL_COLOR = (180, 180, 180)


def m_to_px(x_m: float) -> float:
    return x_m * PIXELS_PER_METER


class MassSpringChain:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.font = pygame.font.SysFont(None, 18)

        # Physikraum
        self.space = pymunk.Space()

        # Laufzeit-Parameter (veränderbar per Tastatur)
        self.gravity_presets = [
            (0.0, 0.0),
            (0.0, 9.81 * PIXELS_PER_METER),
        ]
        self.gravity_index = 0
        self.space.gravity = self.gravity_presets[self.gravity_index]

        self.damping = SPACE_DAMPING
        self.space.damping = self.damping

        self.impulse_time_s = IMPULSE_TIME_S
        self.anchor_lift_m = ANCHOR_LIFT_M

        # Zeitschritt
        self.dt = 1.0 / 120.0
        self.physics_steps_per_frame = 2

        # Aufbau der Kette
        self.bodies: list[pymunk.Body] = []
        self.shapes: list[pymunk.Shape] = []
        self.springs: list[pymunk.DampedSpring] = []

        self._build_chain()

        # Steuerung des Federimpulses
        self.running = True
        self.time_accum = 0.0
        self.impulse_applied = False

    def _build_chain(self) -> None:
        # Geometrie der Kette
        rest_px = m_to_px(L0_M)          # Ruhelänge (entspannt)
        span_px = m_to_px(LV_M)          # Start-Abstand (Vorspannung)
        start_x = 100.0
        baseline_y = HEIGHT * 0.5

        # Linker Anker (Kinematischer Körper, per Sprung bewegbar)
        self.left_anchor = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.left_anchor_base = (start_x, baseline_y)
        self.left_anchor.position = self.left_anchor_base

        # Rechter Anker (Wand) als statischer Körper an rechter Seite
        wall_x = start_x + (N_MASSES + 1) * span_px
        self.right_anchor = self.space.static_body
        self.right_anchor_pos = (wall_x, baseline_y)

        # Optionale Darstellung der Wand (als dünnes Segment)
        self.wall_shape = pymunk.Segment(self.right_anchor, (wall_x, 40), (wall_x, HEIGHT - 40), 1.0)
        self.wall_shape.color = (*WALL_COLOR, 255)
        self.wall_shape.elasticity = 0.0
        self.wall_shape.friction = 0.0
        self.space.add(self.wall_shape)

        # Massen erzeugen (Startpositionen mit Vorspannungsabstand)
        for i in range(N_MASSES):
            x = start_x + (i + 1) * span_px
            y = baseline_y
            moment = pymunk.moment_for_circle(MASS_KG, 0, MASS_RADIUS_PX)
            body = pymunk.Body(MASS_KG, moment)
            body.position = (x, y)
            shape = pymunk.Circle(body, MASS_RADIUS_PX)
            shape.color = (*MASS_COLOR, 255)
            shape.elasticity = 0.0
            shape.friction = 0.0
            self.bodies.append(body)
            self.shapes.append(shape)

        self.space.add(*self.bodies, *self.shapes)

        # Federn anbringen:
        # 1) Erste Feder: linker Anker ↔ erste Masse
        s0 = pymunk.DampedSpring(
            self.left_anchor,
            self.bodies[0],
            (0, 0),
            (0, 0),
            rest_px,
            SPRING_STIFFNESS,
            SPRING_DAMPING,
        )
        self.springs.append(s0)

        # 2) Zwischen benachbarten Massen
        for i in range(1, N_MASSES):
            s = pymunk.DampedSpring(
                self.bodies[i - 1],
                self.bodies[i],
                (0, 0),
                (0, 0),
                rest_px,
                SPRING_STIFFNESS,
                SPRING_DAMPING,
            )
            self.springs.append(s)

        # 3) Letzte Feder: letzte Masse ↔ rechte Wand (statischer Körper)
        s_last = pymunk.DampedSpring(
            self.bodies[-1],
            self.right_anchor,
            (0, 0),
            self.right_anchor_pos,
            rest_px,
            SPRING_STIFFNESS,
            SPRING_DAMPING,
        )
        self.springs.append(s_last)

        self.space.add(*self.springs)

    def _apply_first_spring_impulse(self) -> None:
        # Einmaliger Sprung des linken Ankers nach oben
        lift_px = m_to_px(self.anchor_lift_m)
        x, y = self.left_anchor.position
        self.left_anchor.position = (x, y - lift_px)
        self.impulse_applied = True

    def _process_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_g:
                    # Toggle Gravitation zwischen 0 und ~1g
                    self.gravity_index = (self.gravity_index + 1) % len(self.gravity_presets)
                    self.space.gravity = self.gravity_presets[self.gravity_index]
                elif k == pygame.K_1:
                    # Dämpfung verringern
                    self.damping = max(0.0, self.damping - 0.01)
                    self.space.damping = self.damping
                elif k == pygame.K_2:
                    # Dämpfung erhöhen
                    self.damping = min(0.999, self.damping + 0.01)
                    self.space.damping = self.damping
                elif k == pygame.K_3:
                    # Impuls-Höhe verringern
                    self.anchor_lift_m = max(0.0, self.anchor_lift_m - 0.05)
                elif k == pygame.K_4:
                    # Impuls-Höhe erhöhen
                    self.anchor_lift_m = min(2.0, self.anchor_lift_m + 0.05)
                elif k == pygame.K_5:
                    # Impulszeit früher
                    self.impulse_time_s = max(0.0, self.impulse_time_s - 0.05)
                elif k == pygame.K_6:
                    # Impulszeit später
                    self.impulse_time_s = min(5.0, self.impulse_time_s + 0.05)
                elif k == pygame.K_i:
                    # Impuls sofort auslösen, falls noch nicht erfolgt
                    if not self.impulse_applied:
                        self._apply_first_spring_impulse()
                elif k == pygame.K_r:
                    # Simulation zurücksetzen
                    self._reset_sim()

    def _reset_sim(self) -> None:
        # Raum neu erstellen (einfachste saubere Variante)
        self.space = pymunk.Space()
        self.space.gravity = self.gravity_presets[self.gravity_index]
        self.space.damping = self.damping
        self.bodies = []
        self.shapes = []
        self.springs = []
        self._build_chain()
        self.time_accum = 0.0
        self.impulse_applied = False

    def _draw(self) -> None:
        self.screen.fill(BG_COLOR)

        # Federn als Linien zeichnen (inkl. Enden)
        # Linker Anker → erste Masse
        p0 = self.left_anchor.position
        p1 = self.bodies[0].position
        pygame.draw.line(self.screen, LINE_COLOR, (p0.x, p0.y), (p1.x, p1.y), 1)

        # Zwischen Massen
        for i in range(1, N_MASSES):
            a = self.bodies[i - 1].position
            b = self.bodies[i].position
            pygame.draw.line(self.screen, LINE_COLOR, (a.x, a.y), (b.x, b.y), 1)

        # Letzte Masse → rechte Wand
        last = self.bodies[-1].position
        pygame.draw.line(self.screen, LINE_COLOR, (last.x, last.y), self.right_anchor_pos, 1)

        # Massen als kleine Kreise
        for body in self.bodies:
            pygame.draw.circle(
                self.screen, MASS_COLOR, (int(body.position.x), int(body.position.y)), MASS_RADIUS_PX
            )

        # Optionale Debug-Zeichnung der Shapes (kleine Kreise) und Wand
        # self.space.debug_draw(self.draw_options)

        # HUD
        lines = [
            f"g: {self.space.gravity[1] / PIXELS_PER_METER:0.2f} m/s^2  (g toggeln: G)",
            f"damping: {self.space.damping:0.3f}  (1/2 -/+)",
            f"impulse_time: {self.impulse_time_s:0.2f}s  (5/6 -/+)",
            f"impulse_lift: {self.anchor_lift_m:0.2f} m  (3/4 -/+)",
            "i: Impuls jetzt | r: Reset | Esc: Quit",
        ]
        y = 8
        for txt in lines:
            surf = self.font.render(txt, True, (0, 0, 0))
            self.screen.blit(surf, (8, y))
            y += 18

        pygame.display.flip()

    def run(self) -> None:
        while self.running:
            self._process_events()

            # Impuls der ersten Feder nach IMPULSE_TIME_S einmalig ausführen
            if not self.impulse_applied:
                self.time_accum += self.dt * self.physics_steps_per_frame
                if self.time_accum >= self.impulse_time_s:
                    self._apply_first_spring_impulse()

            # Physik-Integration
            for _ in range(self.physics_steps_per_frame):
                self.space.step(self.dt)

            self._draw()
            self.clock.tick(FPS_LIMIT)
            pygame.display.set_caption(f"fps: {self.clock.get_fps():.1f}")


def main() -> None:
    app = MassSpringChain()
    app.run()


if __name__ == "__main__":
    main()
