package app.botted;

import org.apache.commons.lang3.StringUtils;
import org.jsoup.Jsoup;
import org.jsoup.Connection;
import org.jsoup.Connection.Method;
import org.jsoup.Connection.Response;
import javax.security.sasl.AuthenticationException;
import java.time.Instant;
import java.util.Base64;
import java.io.IOException;
import com.google.gson.*;

public class Reddit {

    /**
     * Protected and private variables
     */
    private final String BASE_URL = "https://www.reddit.com";
    private final String OAUTH_URL = "https://oauth.reddit.com";
    private final String clientId = "GgPNctP2KQdth-iX6aMGUQ";
    private final String clientSecret = "6zov1gDWJ8Ij60yH3L7q6N_LnPUZHA";
    private final String userAgent = "botted 0.0.1";
    private String token;
    private long expirationDate;
    protected String subreddit;

    /**
     * Default constructor
     * @throws IOException
     * @throws InterruptedException
     */
    public Reddit() throws IOException, InterruptedException {
        ensureConnection();
    }

    /**
     * Constructor with parameters
     * @param subreddit The specific subreddit
     * @throws IOException
     * @throws InterruptedException
     */
    public Reddit(String subreddit) throws IOException, InterruptedException {
        this.subreddit = subreddit;
        ensureConnection();
    }

    //getter

    public String getSubreddit() {
        return subreddit;
    }

    //setter

    public void setSubreddit(String subreddit) {
        this.subreddit = subreddit;
    }

    /**
     * Gets access token
     * Generates authorization header
     * Opens the connection and gets response from the server
     * Sets access token and expiration time
     * @throws IOException
     */
    public void connect() throws IOException {
        Connection conn = Jsoup.connect(BASE_URL + "/api/v1/access_token").ignoreContentType(true).ignoreHttpErrors(true).method(Method.POST).userAgent(userAgent);
        conn.data("grant_type", "client_credentials");

        String combination = clientId + ":" + clientSecret;
        combination = Base64.getEncoder().encodeToString(combination.getBytes());
        conn.header("Authorization", "Basic " + combination);

        Response res = conn.execute();
        JsonObject object = JsonParser.parseString(res.body()).getAsJsonObject();

        this.token = object.get("access_token").getAsString();
        this.expirationDate = object.get("expires_in").getAsInt() + Instant.now().getEpochSecond();
    }

    /**
     * Use end point to ensure connection
     * @param endpointPath Path for end point
     * @return The end point
     * @throws IOException
     * @throws InterruptedException
     */
    public JsonObject useEndpoint(String endpointPath) throws IOException, InterruptedException {
        ensureConnection();
        Connection connection = Jsoup.connect(OAUTH_URL + endpointPath);
        connection.header("Authorization", "bearer " + token).ignoreContentType(true).userAgent(userAgent);
        return JsonParser.parseString(connection.execute().body()).getAsJsonObject();
    }

    /**
     * Use end point submission for connection
     * @param endpointPath Path for end point
     * @return The end point submission
     * @throws IOException
     * @throws InterruptedException
     */
    public JsonArray useEndpointSubmission(String endpointPath) throws IOException, InterruptedException {
        ensureConnection();
        Connection connection = Jsoup.connect(OAUTH_URL + endpointPath);
        connection.header("Authorization", "bearer " + token).ignoreContentType(true).userAgent(userAgent);
        return JsonParser.parseString(connection.execute().body()).getAsJsonArray();
    }

    /**
     * Ensure the connection is authenticated
     * @throws IOException
     * @throws InterruptedException
     * @throws AuthenticationException
     */
    public void ensureConnection() throws IOException, InterruptedException, AuthenticationException {
        // There is no token
        if (token == null) {
            connect();
        }
        // The token is expired
        if (Instant.now().getEpochSecond() > expirationDate) {
            connect();
        }
    }

    /**
     * Get username from input validate it
     * @returns username
     */
    public String readInput(String input) throws IOException, InterruptedException {
        String user = "";
        String endpoint = "";
        if (!input.contains("/")) {
            user = input;
        }
        if (input.startsWith("u/") || input.contains("/u/")) {
            String[] e = input.split("u/");
            String[] f = e[1].split("/");
            user = f[0];
        }

        if (input.contains("/user/")) {
            String[] g = input.split("user/");
            String[] h = g[1].split("/");
            user = h[0];
        }

        if (input.contains("/comments/") && !input.contains("/comment/")) {
            String[] c = input.split(".com/");
            String[] d = c[1].split("/");
            if (d.length <= 5) // submission
                endpoint = "/r/" + d[1] + "/comments/" + d[3];
            JsonArray submissionInfo = useEndpointSubmission(endpoint);
            JsonObject array0 = submissionInfo.get(0).getAsJsonObject();
            JsonObject submissionData = (JsonObject) array0.getAsJsonObject().get("data");
            JsonArray submissionChildren = submissionData.getAsJsonArray("children");
            for (JsonElement items : submissionChildren) {
                JsonObject data3 = (JsonObject) items.getAsJsonObject().get("data");
                String author = String.valueOf(data3.getAsJsonObject().get("author"));
                user = author.replace("\"", "");
            }

            if (d.length >= 6) { // comment
                endpoint = "/r/" + d[1] + "/api/info?id=t1_" + d[5];
                JsonObject info = useEndpoint(endpoint);
                JsonObject data = info.getAsJsonObject("data");
                JsonArray children = data.getAsJsonArray("children");
                for (JsonElement item : children) {
                    JsonObject data11 = (JsonObject) item.getAsJsonObject().get("data");
                    String author = String.valueOf(data11.getAsJsonObject().get("author"));
                    user = author.replace("\"", "");
                }
            }
        }

        if (input.contains("/comments/") && input.contains("/comment/")) { // comment
            String[] a = input.split("/r/");
            String[] subreddit = a[1].split("/");
            String[] b = input.split("/comment/");
            String[] commentId = b[1].split("/");
            endpoint = "/r/" + subreddit[0] + "/api/info?id=t1_" + commentId[0];
            JsonObject info = useEndpoint(endpoint);
            JsonObject data = info.getAsJsonObject("data");
            JsonArray children = data.getAsJsonArray("children");
            for (JsonElement item : children) {
                JsonObject data1 = (JsonObject) item.getAsJsonObject().get("data");
                String author = String.valueOf(data1.getAsJsonObject().get("author"));
                user = author.replace("\"", "");
            }
        }
        return user;
    }

    /**
     * Finding similarities between comments (referenced as Strings)
     * @param x String 1 to compare to
     * @param y String 2 to compare to
     * @return
     */
    public static double findSimilarity(String x, String y) {
        double maxLength = Double.max(x.length(), y.length());
        if (maxLength > 0)
            return (maxLength - StringUtils.getLevenshteinDistance(x, y)) / maxLength;
        return 0.0;
    }

    public String random() throws IOException, InterruptedException {
        JsonObject random = useEndpoint("/r/all/comments");
        JsonObject data = (JsonObject) random.get("data");
        JsonArray children = data.getAsJsonArray("children");
        String author = null;
        for (JsonElement item : children) {
            JsonObject dat = (JsonObject) item.getAsJsonObject().get("data");
            author = String.valueOf(dat.getAsJsonObject().get("author"));

        }
        return author.substring(1,author.length()-1);
    }


    /**
     * Send results to string
     * @return BASE_URL, OAUTH_URL, clientId, clientSecret, userAgent, token, expirationDate, and subreddit
     */
    @Override
    public String toString() {
        return "Reddit{" +
                "BASE_URL='" + BASE_URL + '\'' +
                ", OAUTH_URL='" + OAUTH_URL + '\'' +
                ", clientId='" + clientId + '\'' +
                ", clientSecret='" + clientSecret + '\'' +
                ", userAgent='" + userAgent + '\'' +
                ", token='" + token + '\'' +
                ", expirationDate=" + expirationDate +
                ", subreddit='" + subreddit + '\'' +
                '}';
    }

}