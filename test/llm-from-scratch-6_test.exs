defmodule LlmFromScratch6Test do
  use ExUnit.Case

  alias LlmScratch.FineTuneDataLoader

  @tag :download
  test "6.1 loads SMS spam TSV rows and counts labels" do
    tmp_dir = Path.join(System.tmp_dir!(), "llm_scratch_sms_spam_collection")
    data_file_path = Path.expand("SMSSpamCollection.tsv")

    data_file_path =
      FineTuneDataLoader.download_and_unzip_spam_data(
        zip_path: Path.join(tmp_dir, "sms_spam_collection.zip"),
        extracted_path: Path.join(tmp_dir, "sms_spam_collection"),
        data_file_path: data_file_path
      )

    records = FineTuneDataLoader.load_spam_data(data_file_path)
    label_counts = FineTuneDataLoader.label_counts(records)

    assert length(records) == 5_574
    assert label_counts["ham"] == 4_827
    assert label_counts["spam"] == 747

    balanced_records = FineTuneDataLoader.create_balanced_dataset(records)
    balanced_label_counts = FineTuneDataLoader.label_counts(balanced_records)

    assert length(balanced_records) == 1_494
    assert balanced_label_counts["ham"] == 747
    assert balanced_label_counts["spam"] == 747

    mapped_records = FineTuneDataLoader.map_labels_to_ids(balanced_records)
    mapped_label_counts = FineTuneDataLoader.label_counts(mapped_records)

    assert mapped_label_counts[0] == 747
    assert mapped_label_counts[1] == 747

    {train_records, validation_records, test_records} =
      FineTuneDataLoader.random_split(mapped_records, 0.7, 0.1)

    write_csv!("train.csv", train_records)
    write_csv!("validation.csv", validation_records)
    write_csv!("test.csv", test_records)

    assert length(train_records) == 1_045
    assert length(validation_records) == 149
    assert length(test_records) == 300
    assert File.exists?("train.csv")
    assert File.exists?("validation.csv")
    assert File.exists?("test.csv")
  end

  defp write_csv!(path, records) do
    rows =
      records
      |> Enum.map(fn record ->
        [csv_field(record.label), ",", csv_field(record.text), "\n"]
      end)

    File.write!(path, ["Label,Text\n", rows])
  end

  defp csv_field(value) do
    value
    |> to_string()
    |> String.replace("\"", "\"\"")
    |> then(&"\"#{&1}\"")
  end
end
