from typing import ClassVar

import pytest
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

import machine_failure.models.base as base
from machine_failure.models import Model, all_models, get_model, register


class Toy(Model):
    name = "toy"
    role = "candidate"
    search_space: ClassVar = {"model__max_depth": [2, 3]}
    tuned_settings: ClassVar = {"max_depth": 3}

    def make_estimator(self) -> BaseEstimator:
        return DecisionTreeClassifier(max_depth=2)


class Untuned(Model):
    name = "untuned"
    role = "baseline"

    def make_estimator(self) -> BaseEstimator:
        return DummyClassifier()


@pytest.fixture
def empty_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "_REGISTRY", {})


def test_model_is_abstract() -> None:
    with pytest.raises(TypeError):
        Model()  # type: ignore[abstract]


def test_build_puts_the_estimator_behind_preprocessing() -> None:
    pipeline = Toy().build("raw")

    assert [name for name, _ in pipeline.steps] == ["preprocess", "model"]
    assert isinstance(pipeline.named_steps["model"], DecisionTreeClassifier)


def test_build_gives_a_fresh_estimator_each_time() -> None:
    model = Toy()

    assert model.build().named_steps["model"] is not model.build().named_steps["model"]


def test_build_applies_settings_by_bare_name() -> None:
    pipeline = Toy().build(settings={"max_depth": 7})

    assert pipeline.get_params()["model__max_depth"] == 7


def test_build_tuned_applies_the_tuned_settings() -> None:
    assert Toy().build_tuned().get_params()["model__max_depth"] == 3


def test_build_tuned_refuses_an_untuned_model() -> None:
    assert not Untuned().is_tuned
    assert Toy().is_tuned
    with pytest.raises(ValueError, match="no tuned settings"):
        Untuned().build_tuned()


@pytest.mark.usefixtures("empty_registry")
def test_registry_keeps_registration_order_and_filters_by_role() -> None:
    register(Toy)
    register(Untuned)

    assert list(all_models()) == ["toy", "untuned"]
    assert list(all_models("baseline")) == ["untuned"]
    assert isinstance(get_model("toy"), Toy)


@pytest.mark.usefixtures("empty_registry")
def test_registering_a_name_twice_is_an_error() -> None:
    register(Toy)

    with pytest.raises(ValueError, match="already registered"):
        register(Toy)


@pytest.mark.usefixtures("empty_registry")
def test_unknown_model_lists_the_known_ones() -> None:
    register(Toy)

    with pytest.raises(KeyError, match="toy"):
        get_model("nope")
