from assistant import config


def test_gigakey_alias_and_bundled_ca(monkeypatch):
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    monkeypatch.delenv("GIGACHAT_CA_BUNDLE", raising=False)
    monkeypatch.setenv("GIGAKEY", "secret")
    monkeypatch.setenv("LLM_PROVIDER", "")
    s = config.Settings()
    assert s.gigachat_credentials == "secret"
    assert s.resolved_llm_provider == "gigachat"
    assert s.gigachat_ca_bundle.endswith("russian_trusted_ca.pem")


def test_env_file_does_not_override_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO_TEST=from_file\nBAR_TEST='quoted'\n# comment\n", encoding="utf-8")
    monkeypatch.setenv("FOO_TEST", "from_env")
    monkeypatch.delenv("BAR_TEST", raising=False)
    config.load_env_file(env)
    import os
    assert os.environ["FOO_TEST"] == "from_env"
    assert os.environ["BAR_TEST"] == "quoted"
    monkeypatch.delenv("BAR_TEST")
