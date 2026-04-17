from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import Cocktail, Ingredient
from app.db.crud import get_all_recipes_with_ingredients


def _ingredient_name(ingredient: Ingredient) -> str:
    return getattr(ingredient, "ingredients_name", None) or getattr(ingredient, "name_kr", "")


def generate_recipe_snapshot(
    db: Session,
    cocktail_id: int,
    volume_ml: int = 90,
) -> dict:
    cocktail = db.query(Cocktail).filter(Cocktail.cocktail_id == cocktail_id).first()
    if not cocktail:
        raise ValueError("cocktail not found")

    all_ri = get_all_recipes_with_ingredients(db)
    rows = all_ri.get(cocktail_id, [])

    total_original = sum(float(recipe.amount_ml or 0) for recipe, _ in rows)
    scale = (volume_ml / total_original) if total_original > 0 else 1.0

    recipe_items = []
    for recipe, ingredient in sorted(rows, key=lambda x: x[0].step_order):
        scaled_ml = round(float(recipe.amount_ml or 0) * scale, 1)
        recipe_items.append(
            {
                "ingredient_id": ingredient.ingredient_id,
                "ingredient_name": _ingredient_name(ingredient),
                "amount_ml": scaled_ml,
                "step_order": recipe.step_order,
                "is_optional": bool(recipe.is_optional),
            }
        )

    return {
        "cocktail_id": cocktail.cocktail_id,
        "cocktail_name": cocktail.name_kr,
        "total_volume_ml": volume_ml,
        "recipe": recipe_items,
    }


def generate_output_json(
    db: Session,
    cocktail_id: int,
    volume_ml: int = 90,
) -> dict:
    snapshot = generate_recipe_snapshot(db, cocktail_id, volume_ml)

    return {
        "cocktail_id": snapshot["cocktail_id"],
        "cocktail_name": snapshot["cocktail_name"],
        "total_volume_ml": snapshot["total_volume_ml"],
        "steps": snapshot["recipe"],
    }