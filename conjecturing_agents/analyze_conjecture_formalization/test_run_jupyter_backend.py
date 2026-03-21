from conjecturing_agents.tool_calling_backends.jupyter import JupyterKernelConfig, JupyterToolBackend

def main() -> None:
    backend = JupyterToolBackend(
        description="Test python tool",
        cfg=JupyterKernelConfig(
            timeout_seconds=6.0,
            init_on_create=False,
        ),
    )

    try:
        tests = [
            ("simple arithmetic", "2 + 3"),
            ("state persists", "x = 10"),
            ("state readback", "x * 7"),
            ("imports available", "import sympy as sp\nx = sp.Symbol('x')\nsp.expand((x + 1)**3)"),
            ("timeout", "while True:\n    pass"),
        ]

        for name, code in tests:
            print(f"\n=== {name} ===")
            result = backend.execute(code)
            print("success   :", result.success)
            print("timed_out :", result.timed_out)
            print("had_error :", result.had_error)
            print("elapsed_ms:", result.elapsed_ms)
            print("output:")
            print(result.output)

        print("\n=== reset ===")
        backend.reset()
        result = backend.execute("globals().get('x', 'missing')")
        print(result.output)

    finally:
        backend.close()

if __name__ == "__main__":
    main()