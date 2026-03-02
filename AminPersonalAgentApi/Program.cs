using System.Net.Http.Json;
using Microsoft.Extensions.Options;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();
builder.Services.AddCors(options =>
{
    options.AddPolicy("AngularDev", policy =>
    {
        policy
            .WithOrigins("http://localhost:4200", "http://127.0.0.1:4200")
            .AllowAnyHeader()
            .AllowAnyMethod();
    });
});
builder.Services.Configure<RagServiceOptions>(builder.Configuration.GetSection("RagService"));
builder.Services.AddHttpClient("RagService", (serviceProvider, client) =>
{
    var options = serviceProvider.GetRequiredService<IOptions<RagServiceOptions>>().Value;
    client.BaseAddress = new Uri(options.BaseUrl);
    client.Timeout = TimeSpan.FromSeconds(options.TimeoutSeconds);
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseHttpsRedirection();
app.UseCors("AngularDev");

app.MapGet("/health", async (IHttpClientFactory httpClientFactory, CancellationToken cancellationToken) =>
{
    try
    {
        var client = httpClientFactory.CreateClient("RagService");
        var ragHealth = await client.GetFromJsonAsync<object>("/health", cancellationToken);
        return Results.Ok(new
        {
            status = "ok",
            api = "AminPersonalAgentApi",
            rag = ragHealth
        });
    }
    catch (Exception exception)
    {
        return Results.Ok(new
        {
            status = "degraded",
            api = "AminPersonalAgentApi",
            ragError = exception.Message
        });
    }
})
.WithName("Health");

app.MapPost("/api/chat", async (
    ChatRequest request,
    IHttpClientFactory httpClientFactory,
    IOptions<RagServiceOptions> ragOptions,
    CancellationToken cancellationToken) =>
{
    if (string.IsNullOrWhiteSpace(request.Message))
    {
        return Results.BadRequest(new { error = "message is required" });
    }

    var options = ragOptions.Value;
    var payload = new RagChatRequest(
        message: request.Message,
        session_id: request.SessionId ?? options.DefaultSessionId,
        history_turns: request.HistoryTurns ?? options.DefaultHistoryTurns,
        top_k: request.TopK ?? options.DefaultTopK,
        llm_provider: request.LlmProvider ?? options.DefaultLlmProvider,
        llm_model: request.LlmModel ?? options.DefaultLlmModel,
        llm_url: request.LlmUrl ?? options.DefaultLlmUrl,
        api_key: request.ApiKey ?? string.Empty,
        temperature: request.Temperature ?? options.DefaultTemperature,
        facts_only: request.FactsOnly ?? false,
        debug_prompt: request.DebugPrompt ?? false
    );

    var client = httpClientFactory.CreateClient("RagService");

    HttpResponseMessage response;
    try
    {
        response = await client.PostAsJsonAsync("/chat", payload, cancellationToken);
    }
    catch (Exception exception)
    {
        return Results.Problem(
            title: "Failed to reach internal RAG service",
            detail: exception.Message,
            statusCode: StatusCodes.Status503ServiceUnavailable);
    }

    var bodyText = await response.Content.ReadAsStringAsync(cancellationToken);
    if (!response.IsSuccessStatusCode)
    {
        return Results.Problem(
            title: "Internal RAG service returned an error",
            detail: bodyText,
            statusCode: (int)response.StatusCode);
    }

    return Results.Content(bodyText, "application/json");
})
.WithName("Chat");

app.Run();

public sealed record ChatRequest(
    string Message,
    string? SessionId,
    int? HistoryTurns,
    int? TopK,
    string? LlmProvider,
    string? LlmModel,
    string? LlmUrl,
    string? ApiKey,
    double? Temperature,
    bool? FactsOnly,
    bool? DebugPrompt
);

public sealed record RagChatRequest(
    string message,
    string session_id,
    int history_turns,
    int top_k,
    string llm_provider,
    string llm_model,
    string llm_url,
    string api_key,
    double temperature,
    bool facts_only,
    bool debug_prompt
);

public sealed class RagServiceOptions
{
    public string BaseUrl { get; set; } = "http://127.0.0.1:8091";
    public int TimeoutSeconds { get; set; } = 180;
    public string DefaultSessionId { get; set; } = "amin_about_me";
    public int DefaultHistoryTurns { get; set; } = 4;
    public int DefaultTopK { get; set; } = 3;
    public string DefaultLlmProvider { get; set; } = "openai-compatible";
    public string DefaultLlmModel { get; set; } = "llama";
    public string DefaultLlmUrl { get; set; } = "http://127.0.0.1:8080";
    public double DefaultTemperature { get; set; } = 0.2;
}
