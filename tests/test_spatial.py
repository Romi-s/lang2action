from lang2action.perception import SceneObject, infer_relations


def obj(id: str, x: float, y: float, z: float = 0.02) -> SceneObject:
    return SceneObject(id=id, category="cube", color="gray", position=(x, y, z))


def rel_set(objects):
    return {(r.subject_id, r.relation, r.object_id) for r in infer_relations(objects)}


def test_left_right():
    rels = rel_set([obj("a", 0.0, 0.0), obj("b", 0.2, 0.0)])
    assert ("a", "left_of", "b") in rels
    assert ("b", "right_of", "a") in rels


def test_front_behind():
    # +y is away from the viewer: smaller y = in front
    rels = rel_set([obj("a", 0.0, 0.0), obj("b", 0.0, 0.2)])
    assert ("a", "in_front_of", "b") in rels
    assert ("b", "behind", "a") in rels


def test_stacked():
    rels = rel_set([obj("top", 0.0, 0.0, 0.06), obj("base", 0.0, 0.0, 0.02)])
    assert ("top", "on_top_of", "base") in rels
    # a stacked object gets no planar relation to its base
    assert not any(s == "top" and o == "base" and r != "on_top_of" for s, r, o in rels)


def test_near_aligned_objects_generate_no_noise():
    # 1 cm apart, below the 3 cm gap: no left/right assertion either way
    rels = rel_set([obj("a", 0.0, 0.0), obj("b", 0.01, 0.0)])
    assert rels == set()


def test_diagonal_pair_gets_both_axes():
    rels = rel_set([obj("a", 0.0, 0.0), obj("b", 0.2, 0.2)])
    assert ("a", "left_of", "b") in rels
    assert ("a", "in_front_of", "b") in rels
