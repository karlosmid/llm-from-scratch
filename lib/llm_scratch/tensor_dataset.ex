defprotocol LlmScratch.Dataset do
  def get_item(dataset, index)
  def length(dataset)
end

defmodule LlmScratch.TensorDataset do
  defstruct [:data, :labels]

  defimpl LlmScratch.Dataset do
    def get_item(%{data: data, labels: labels}, index) do
      {Enum.at(data, index), Enum.at(labels, index)}
    end

    def length(%{data: data}) do
      Enum.count(data)
    end
  end
end
