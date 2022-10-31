try:
    import multiscale_run.metabolism_utils as mu

    def test_gen_metabolism_model():
        mu.gen_metabolism_model()

except ModuleNotFoundError:
    pass
