# -*- coding: utf-8 -*-
"""
scene_objects.py
ВАЖНАЯ РЕВИЗИЯ: убрано любое деление объектов на "опасность"/"инструмент" --
это было мнимое знание, которого у Viktim быть не может (в оригинале он не
получает подсказок "это плохо, это хорошо"). Остались только ФИЗИЧЕСКИЕ
СВОЙСТВА объектов -- это законы симулируемого мира (как реальный ластик
физически стирает то, что под ним, независимо от того, "хочет" ли этого
кто-то), а не подсказки для мозга Viktim. Мозг видит только сырые
последствия (удар, сдавливание, освобождение) через world_physics.py --
и должен сам, через боль/облегчение и пластичность, разобраться, что
безопасно, а что нет.

Физические свойства (PRESETS):
  mass         -- масса (участвует в силе удара при столкновении)
  radius       -- радиус объекта
  is_entangling-- физически спутывает движение при контакте (как канат/лассо)
  erase_radius -- если >0, объект физически стирает всё, что попадает в этот
                  радиус ВОКРУГ НЕГО (то же самое произошло бы, будь этот
                  объект хоть в руках Viktim, хоть просто лежи он на полу и
                  чего-то коснись) -- это чистая физика материала, а не флаг
                  "оружие"
  burns_through-- физически способен пережечь/разрушить спутывающий объект
                  при контакте (как лупа фокусирует свет и пережигает верёвку)
  grabbable    -- достаточно лёгкий и небольшой, чтобы тело могло его схватить
                  при простом контакте (хватательный рефлекс, а не решение)
  gravity      -- падает вниз (тяжёлые объекты вроде наковальни)
  vx, vy       -- текущая скорость (для летящих снарядов)
"""

PRESETS = {
    "anvil":    dict(mass=40.0, radius=17, is_entangling=False, erase_radius=0.0,
                      burns_through=False, grabbable=False, gravity=True),
    "shuriken": dict(mass=2.0, radius=8, is_entangling=False, erase_radius=0.0,
                      burns_through=False, grabbable=False, gravity=False),
    "lasso":    dict(mass=1.0, radius=14, is_entangling=True, erase_radius=0.0,
                      burns_through=False, grabbable=False, gravity=False),
    # --- оружие, которое сам Алан рисовал Pencil-инструментом против Victim ---
    "plasma_gun":   dict(mass=3.0, radius=10, is_entangling=False, erase_radius=0.0,
                          burns_through=False, grabbable=False, gravity=False),
    "frame_cannon": dict(mass=6.0, radius=12, is_entangling=False, erase_radius=0.0,
                          burns_through=False, grabbable=False, gravity=False),
    "sword":    dict(mass=1.5, radius=9, is_entangling=False, erase_radius=0.0,
                      burns_through=False, grabbable=True, gravity=False),
    "paintbrush": dict(mass=0.3, radius=8, is_entangling=False, erase_radius=0.0,
                      burns_through=False, grabbable=True, gravity=False),
    "eraser":   dict(mass=0.3, radius=8, is_entangling=False, erase_radius=26.0,
                      burns_through=False, grabbable=True, gravity=False),
    "pen":      dict(mass=0.2, radius=6, is_entangling=False, erase_radius=0.0,
                      burns_through=False, grabbable=True, gravity=False),
    "magnifying_glass": dict(mass=0.3, radius=9, is_entangling=False, erase_radius=0.0,
                              burns_through=True, grabbable=True, gravity=False),
    # честно помечено: это НЕ из оригинального мультфильма, песочные дополнения
    "ice":      dict(mass=0.0, radius=14, is_entangling=False, erase_radius=0.0,
                      burns_through=False, grabbable=False, gravity=False, freezes=True),
    "portal":   dict(mass=0.0, radius=14, is_entangling=False, erase_radius=0.0,
                      burns_through=False, grabbable=False, gravity=False, teleports=True),

    # --- обломки самого интерфейса: Victim ломает Adobe и строит из этого щиты/стены
    #     (в оригинале он тоже прорывается сквозь кадры таймлайна и утаскивает их с собой) ---
    "timeline_frame":   dict(mass=3.0, radius=13, is_entangling=False, erase_radius=0.0,
                              burns_through=False, grabbable=True, gravity=False),
    "toolbar_fragment":  dict(mass=2.0, radius=11, is_entangling=False, erase_radius=0.0,
                               burns_through=False, grabbable=True, gravity=False),
}

# цвета -- чисто для отрисовки, не несут смысловой нагрузки "хорошо/плохо"
COLORS = {
    "anvil": (0.35, 0.35, 0.4, 1), "shuriken": (0.5, 0.5, 0.55, 1),
    "lasso": (0.8, 0.6, 0.2, 1), "paintbrush": (0.2, 0.5, 0.9, 1),
    "eraser": (0.9, 0.9, 0.9, 1), "pen": (0.15, 0.15, 0.15, 1),
    "magnifying_glass": (0.7, 0.85, 0.95, 1), "ice": (0.6, 0.9, 1.0, 1),
    "portal": (0.6, 0.2, 0.9, 1), "plasma_gun": (0.2, 0.9, 0.9, 1),
    "frame_cannon": (0.4, 0.3, 0.2, 1), "sword": (0.75, 0.75, 0.8, 1),
    "timeline_frame": (0.85, 0.85, 0.3, 1), "toolbar_fragment": (0.6, 0.6, 0.65, 1),
}

PALETTE = [
    (0.05, 0.05, 0.05, 1), (0.8, 0.1, 0.1, 1), (0.1, 0.5, 0.9, 1),
    (0.1, 0.7, 0.2, 1), (0.9, 0.7, 0.1, 1), (0.6, 0.2, 0.8, 1),
]


PROJECTILE_TAGS = {"shuriken", "plasma_gun", "frame_cannon"}


class SceneObject:
    """Физический объект сцены. Только измеримые физические свойства, без ярлыков."""
    __slots__ = ("x", "y", "tag", "radius", "mass", "is_entangling", "erase_radius",
                 "burns_through", "grabbable", "gravity", "freezes", "teleports",
                 "vx", "vy", "held_by")

    def __init__(self, x, y, tag, vx=0.0, vy=0.0):
        p = PRESETS.get(tag, dict(mass=1.0, radius=10, is_entangling=False,
                                    erase_radius=0.0, burns_through=False,
                                    grabbable=False, gravity=False))
        self.x, self.y, self.tag = x, y, tag
        self.radius = p["radius"]
        self.mass = p["mass"]
        self.is_entangling = p["is_entangling"]
        self.erase_radius = p["erase_radius"]
        self.burns_through = p["burns_through"]
        self.grabbable = p["grabbable"]
        self.gravity = p["gravity"]
        self.freezes = p.get("freezes", False)
        self.teleports = p.get("teleports", False)
        self.vx, self.vy = vx, vy
        self.held_by = None  # ссылка на Viktim, который держит объект, либо None

    def to_dict(self):
        return {"x": self.x, "y": self.y, "tag": self.tag, "vx": self.vx, "vy": self.vy}

    @staticmethod
    def from_dict(d):
        return SceneObject(d["x"], d["y"], d["tag"], d.get("vx", 0.0), d.get("vy", 0.0))
