"""Check import cycles and dependency boundaries in the package."""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

FORBIDDEN_IMPORTS: dict[str, set[str]] = {
    "geoparser.db": {
        "geoparser.context",
        "geoparser.modules",
        "geoparser.project",
        "geoparser.services",
    },
    "geoparser.modules": {
        "geoparser.context",
        "geoparser.db",
        "geoparser.project",
        "geoparser.services",
    },
    "geoparser.context": {
        "geoparser.modules",
        "geoparser.project",
        "geoparser.services",
    },
    "geoparser.services": {
        "geoparser.context",
        "geoparser.modules",
        "geoparser.project",
    },
}


# Modules that must stay pure domain logic: they may import the standard
# library and nothing else. Keeping a module on this list is what lets it be
# tested and reasoned about on its own, rather than through whatever heavy
# collaborators its callers happen to construct.
PURE_MODULES: set[str] = {
    "geoparser.evaluation",
    "geoparser.modules.resolvers.context",
    "geoparser.modules.resolvers.ranking",
}


def _module_name(path: Path, package_root: Path, package_name: str) -> str:
    relative = path.relative_to(package_root).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((package_name, *parts))


def _is_type_checking_test(test: ast.expr) -> bool:
    """Return whether an ``if`` condition is a TYPE_CHECKING guard."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


class _ImportVisitor(ast.NodeVisitor):
    """Collect imports while excluding annotation-only imports."""

    def __init__(self) -> None:
        self.runtime: list[tuple[str | None, int, str | None, tuple[str, ...]]] = []

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_test(node.test):
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.runtime.append((alias.name, node.lineno, None, ()))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.runtime.append(
            (
                node.module,
                node.lineno,
                "." * node.level,
                tuple(alias.name for alias in node.names),
            )
        )


def _resolve_from_import(
    current: str,
    module: str | None,
    dots: str,
    names: tuple[str, ...],
    packages: set[str],
) -> Iterable[str]:
    """Yield import targets that correspond to real internal modules."""
    base = _import_base(current, module, dots, packages)
    yield from _known_import_targets(base, names, packages)


def _import_base(
    current: str, module: str | None, dots: str, packages: set[str]
) -> str:
    """Resolve the absolute or relative base module for an import record."""
    if dots == "":
        return module or ""
    return _relative_import_base(current, module, dots, packages)


def _known_import_targets(
    base: str, names: tuple[str, ...], packages: set[str]
) -> Iterable[str]:
    """Yield imported package modules named by a resolved base."""
    for name in names:
        target = _known_import_target(base, name, packages)
        if target is not None:
            yield target


def _known_import_target(base: str, name: str, packages: set[str]) -> str | None:
    """Resolve one ``from base import name`` to an existing module."""
    exact = f"{base}.{name}" if base else name
    target = exact if exact in packages else base
    return target if target in packages else None


def _relative_import_base(
    current: str, module: str | None, dots: str, packages: set[str]
) -> str:
    """Resolve the base name for a relative import from one module."""
    level = len(dots)
    current_parts = current.split(".")
    parent_parts = current_parts if current in packages else current_parts[:-1]
    base = ".".join(parent_parts[: len(parent_parts) - level + 1])
    if module:
        return f"{base}.{module}" if base else module
    return base


def _is_internal_import(
    imported: str | None, package_name: str, packages: set[str]
) -> bool:
    """Check whether a plain import names a package module in the graph."""
    return bool(
        imported and imported in packages and _is_package_target(imported, package_name)
    )


def _is_package_target(target: str, package_name: str) -> bool:
    """Whether a fully resolved module name belongs to the package."""
    return target == package_name or target.startswith(f"{package_name}.")


def _targets_for_import(
    current: str,
    imported: str | None,
    dots: str | None,
    names: tuple[str, ...],
    package_name: str,
    packages: set[str],
) -> set[str]:
    """Resolve one visitor record to internal package edges."""
    if dots is None:
        if imported is None:
            return set()
        return (
            {imported}
            if _is_internal_import(imported, package_name, packages)
            else set()
        )
    return _relative_targets(current, imported, dots, names, package_name, packages)


def _relative_targets(
    current: str,
    imported: str | None,
    dots: str,
    names: tuple[str, ...],
    package_name: str,
    packages: set[str],
) -> set[str]:
    """Keep resolved relative imports that remain within this package."""
    return {
        target
        for target in _resolve_from_import(current, imported, dots, names, packages)
        if target == package_name or target.startswith(f"{package_name}.")
    }


def _module_imports(
    module: str, path: Path, package_name: str, packages: set[str]
) -> set[str]:
    """Collect the internal runtime imports for one package module."""
    visitor = _ImportVisitor()
    visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    targets: set[str] = set()
    for imported, _lineno, dots, names in visitor.runtime:
        targets.update(
            _targets_for_import(module, imported, dots, names, package_name, packages)
        )
    return targets


def build_import_graph(package_root: Path, package_name: str) -> dict[str, set[str]]:
    """Build a deterministic runtime import graph for one Python package."""
    package_root = package_root.resolve()
    files = sorted(package_root.rglob("*.py"))
    modules = {_module_name(path, package_root, package_name): path for path in files}
    packages = set(modules)
    return {
        module: _module_imports(module, path, package_name, packages)
        for module, path in modules.items()
    }


def find_cycles(graph: Mapping[str, set[str]]) -> list[tuple[str, ...]]:
    """Return deterministic cycles, with the repeated start node at the end."""
    cycles: set[tuple[str, ...]] = set()
    visited: set[str] = set()
    stack: list[str] = []
    positions: dict[str, int] = {}

    def visit(node: str) -> None:
        if node in positions:
            start = positions[node]
            cycles.add((*stack[start:], node))
            return
        if node in visited:
            return
        positions[node] = len(stack)
        stack.append(node)
        for target in sorted(graph.get(node, set())):
            visit(target)
        stack.pop()
        positions.pop(node)
        visited.add(node)

    for node in sorted(graph):
        visit(node)
    return sorted(cycles)


def _within_boundary(source: str, boundary: str) -> bool:
    """Whether a source module belongs to a forbidden-boundary package."""
    return source == boundary or source.startswith(f"{boundary}.")


def _crosses_boundary(target: str, blocked: set[str]) -> bool:
    """Whether one target module is within a blocked dependency family."""
    return any(
        target == prefix or target.startswith(f"{prefix}.") for prefix in blocked
    )


def _violations_for_source(
    source: str, targets: set[str], forbidden: Mapping[str, set[str]]
) -> set[tuple[str, str]]:
    """Collect blocked dependency edges originating from one module."""
    violations = set()
    for boundary, blocked in forbidden.items():
        if _within_boundary(source, boundary):
            violations.update(
                (source, target)
                for target in targets
                if _crosses_boundary(target, blocked)
            )
    return violations


def find_boundary_violations(
    graph: Mapping[str, set[str]],
    forbidden: Mapping[str, set[str]],
) -> list[tuple[str, str]]:
    """Return runtime edges that cross a forbidden package boundary."""
    return sorted(
        {
            violation
            for source, targets in graph.items()
            for violation in _violations_for_source(source, targets, forbidden)
        }
    )


def find_impure_modules(
    package_root: Path, package_name: str, pure_modules: set[str]
) -> list[tuple[str, str]]:
    """
    Find imports that break a module's promise to depend only on the stdlib.

    Args:
        package_root: Directory holding the package's modules
        package_name: Importable name of the package
        pure_modules: Names of the modules that must stay pure

    Returns:
        (module, offending import) pairs, sorted, empty when all are pure
    """
    impure: list[tuple[str, str]] = []
    for path in sorted(package_root.rglob("*.py")):
        module = _module_name(path, package_root, package_name)
        if module not in pure_modules:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        visitor = _ImportVisitor()
        visitor.visit(tree)
        for target, _lineno, dots, _names in visitor.runtime:
            offender = _impure_target(package_name, target, dots)
            if offender is not None:
                impure.append((module, offender))
    return sorted(impure)


def _impure_target(
    package_name: str, target: str | None, dots: str | None
) -> str | None:
    """
    The import that makes a module impure, if this import does.

    Args:
        package_name: Importable name of the package
        target: The imported module, as written
        dots: Leading dots for a relative import, or None for a plain import

    Returns:
        The offending module name, or None when the import is standard library
    """
    if dots:
        return _relative_impure_target(package_name, target)
    if target is None:
        return None
    if _is_package_target(target, package_name):
        return target
    return None if _is_stdlib_import(target) else target


def _relative_impure_target(package_name: str, target: str | None) -> str:
    """Resolve a relative import to the package path it references."""
    suffix = f".{target}" if target else ""
    return f"{package_name}{suffix}"


def _is_stdlib_import(target: str) -> bool:
    """Whether a plain import is rooted in the Python standard library."""
    return target.split(".", maxsplit=1)[0] in sys.stdlib_module_names


def _print_cycles(cycles: list[tuple[str, ...]]) -> None:
    """Print cycle diagnostics in stable path order."""
    if cycles:
        print("Import cycles:")
        for cycle in cycles:
            print(f"  {' -> '.join(cycle)}")


def _print_edges(title: str, edges: list[tuple[str, str]]) -> None:
    """Print a titled list of module dependency edges."""
    if edges:
        print(f"{title}:")
        for source, target in edges:
            print(f"  {source} -> {target}")


def _print_modules(title: str, modules: list[tuple[str, str]]) -> None:
    """Print a titled list of module and forbidden dependency pairs."""
    if modules:
        print(f"{title}:")
        for module, target in modules:
            print(f"  {module} -> {target}")


def main(argv: list[str] | None = None) -> int:
    """Check the package and return non-zero when its architecture is invalid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path("geoparser"))
    args = parser.parse_args(argv)

    package_root = args.package.resolve()
    package_name = package_root.name
    graph = build_import_graph(package_root, package_name)
    cycles = find_cycles(graph)
    violations = find_boundary_violations(graph, FORBIDDEN_IMPORTS)
    impure = find_impure_modules(package_root, package_name, PURE_MODULES)

    _print_cycles(cycles)
    _print_edges("Forbidden dependency edges", violations)
    _print_modules("Pure modules with a forbidden dependency", impure)
    if cycles or violations or impure:
        return 1

    print(f"Architecture checks passed for {package_name} ({len(graph)} modules).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
