defmodule Axon.Data do
  @column_names %{
    "Bag Distance" => 0,
    "Chunk Set" => 1,
    "Dice Sorensen" => 2,
    "Metaphone" => 3,
    "Double Metaphone" => 4,
    "Double Metaphone Chunks" => 5,
    "Jaccard" => 6,
    "Jaro-Winkler" => 7,
    "Levenshtein" => 8,
    "NGram" => 9,
    "Overlap" => 10,
    "Sorted Chunks" => 11,
    "Tversky" => 12,
    "Match" => 13
  }

  def split_inputs(filename, opts \\ []) do
    input_columns = Keyword.fetch!(opts, :input_columns)
    target_column = Keyword.fetch!(opts, :target_column)
    test_train_ratio = Keyword.fetch!(opts, :test_train_ratio)

    parsed =
      filename
      |> File.stream!()
      |> Enum.map(&convert/1)
      |> Enum.reject(&is_nil/1)

    {test_inputs, train_inputs} =
      parsed
      |> slice_columns(input_columns)
      |> Enum.map(fn i -> Nx.tensor([i]) end)
      |> split(test_train_ratio)

    {test_targets, train_targets} =
      parsed
      |> slice_columns([target_column])
      |> Enum.map(fn a -> Nx.tensor([a]) end)
      |> split(test_train_ratio)

    {
      test_inputs,
      test_targets,
      train_inputs,
      train_targets
    }
  end

  defp split(rows, ratio) do
    count = Enum.count(rows)
    Enum.split(rows, ceil(count * ratio))
  end

  def convert(line) do
    [
      bag_distance,
      chunk_set,
      dice_sorensen,
      metaphone,
      double_metaphone,
      double_metaphone_chunks,
      jaccard,
      jaro_winkler,
      levenshtein,
      ngram,
      overlap,
      sorted_chunks,
      tversky,
      match
    ] = line |> String.replace("\n", "") |> String.split("\t")

    [
      bag_distance |> String.trim() |> String.to_float(),
      chunk_set |> String.trim() |> String.to_float(),
      dice_sorensen |> String.trim() |> String.to_float(),
      metaphone |> String.trim() |> String.to_float(),
      double_metaphone |> String.trim() |> String.to_float(),
      double_metaphone_chunks |> String.trim() |> String.to_float(),
      jaccard |> String.trim() |> String.to_float(),
      jaro_winkler |> String.trim() |> String.to_float(),
      levenshtein |> String.trim() |> String.to_float(),
      ngram |> String.trim() |> String.to_float(),
      overlap |> String.trim() |> String.to_float(),
      sorted_chunks |> String.trim() |> String.to_float(),
      tversky |> String.trim() |> String.to_float(),
      match |> String.trim() |> String.to_integer()
    ]
  rescue
    e in ArgumentError ->
      unless e.message =~ "Received unexpected data type which could not be converted." do
        raise e
      end
  end

  defp slice_columns(data, columns) do
    column_indexes =
      columns
      |> Enum.map(fn name -> Map.get(@column_names, name) end)
      |> Enum.reverse()

    data
    |> Stream.map(fn row ->
      column_indexes
      |> Enum.reduce([], fn index, out ->
        value = Enum.at(row, index)
        [value | out]
      end)
    end)
  end
end
