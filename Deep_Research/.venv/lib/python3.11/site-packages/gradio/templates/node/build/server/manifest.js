const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {"start":"_app/immutable/entry/start.DJv2Cs3S.js","app":"_app/immutable/entry/app.XkwneLUg.js","imports":["_app/immutable/entry/start.DJv2Cs3S.js","_app/immutable/chunks/client.Cd1aarwx.js","_app/immutable/entry/app.XkwneLUg.js","_app/immutable/chunks/preload-helper.D6kgxu3v.js"],"stylesheets":[],"fonts":[],"uses_env_dynamic_public":false},
		nodes: [
			__memo(() => import('./chunks/0-C5IPoxrR.js')),
			__memo(() => import('./chunks/1-D1wkpD0I.js')),
			__memo(() => import('./chunks/2-DJbI4FWc.js').then(function (n) { return n.aE; }))
		],
		routes: [
			{
				id: "/[...catchall]",
				pattern: /^(?:\/(.*))?\/?$/,
				params: [{"name":"catchall","optional":false,"rest":true,"chained":true}],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();

const prerendered = new Set([]);

const base = "";

export { base, manifest, prerendered };
//# sourceMappingURL=manifest.js.map
