try:
    import multiscale_run.metabolism_utils as mu
    from julia import Main

    def test_gen_metabolism_model():
        mu.load_metabolism_data(Main)
        mu.gen_metabolism_model(Main)


except ModuleNotFoundError:
    pass
