defmodule AkinMLTest do
  use ExUnit.Case
  doctest AkinML

  test "receive a heartbeat" do
    assert AkinML.heartbeat() == :beat
  end
end
